import logging
import re
from typing import Any, Literal

import ray
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from pydantic import Field
from ray.util.actor_pool import ActorPool

from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field, silence

logger = logging.getLogger(__name__)

# Use the same PII defaults as the original implementation
PII_DEFAULTS = {
    "en": {
        "CREDIT_CARD": "4242 4242 4242 4242",  # Visa testing number
        "IP_ADDRESS": "192.0.2.255",  # RFC 5737 Test Block 1
        "EMAIL_ADDRESS": "name@example.com",  # example.com is intended as blackhole
        "PHONE_NUMBER": "+1 123 456 7890",  # invalid number with correct format
        "IBAN_CODE": "GB29 NWBK60 1613 3192 6819",  # test number
        "URL": "https://www.example.com",
    },
    "de": {
        "CREDIT_CARD": "4242 4242 4242 4242",  # Visa testing number
        "IP_ADDRESS": "192.0.2.255",  # RFC 5737 Test Block 1
        "EMAIL_ADDRESS": "name@beispiel.de",  # beispiel.de is intended as blackhole
        "PHONE_NUMBER": "+49 123 45678910",  # invalid number with correct format
        "IBAN_CODE": "DE02 1203 0000 0000 2020 51",  # test number
        "URL": "https://www.beispiel.de",
    },
}

# AD2!n4!n4!n12!c
IBAN_AD = r"AD\d{2}\s?\d{4}\s?\d{4}\s?[a-zA-Z0-9]{12}"

# AE2!n3!n16!n
IBAN_AE = r"AE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}"

# AL2!n8!n16!c
IBAN_AL = r"AL\d{2}\s?\d{4}\s?\d{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# AT2!n5!n11!n
IBAN_AT = r"AT\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# AZ2!n4!a20!c
IBAN_AZ = r"AZ\d{2}\s?[A-Z]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# BA2!n3!n3!n8!n2!n
IBAN_BA = r"BA\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# BE2!n3!n7!n2!n
IBAN_BE = r"BE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}"

# BG2!n4!a4!n2!n8!c
IBAN_BG = r"BG\d{2}\s?[A-Z]{4}\s?\d{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# BH2!n4!a14!c
IBAN_BH = r"BH\d{2}\s?[A-Z]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{2}"

# BI2!n5!n5!n11!n2!n
IBAN_BI = r"BI\d{2}\s?\d{5}\s?\d{5}\s?\d{11}\s?\d{2}"

# BR2!n8!n5!n10!n1!a1!c
IBAN_BR = r"BR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}[A-Z]\s?[a-zA-Z0-9]"

# BY2!n4!c4!n16!c
IBAN_BY = r"BY\d{2}\s?[a-zA-Z0-9]{4}\s?\d{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# CH2!n5!n12!c
IBAN_CH = r"CH\d{2}\s?\d{4}\s?\d[a-zA-Z0-9]{3}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]"

# CR2!n4!n14!n
IBAN_CR = r"CR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# CY2!n3!n5!n16!c
IBAN_CY = r"CY\d{2}\s?\d{4}\s?\d{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# CZ2!n4!n6!n10!n
IBAN_CZ = r"CZ\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# DE2!n8!n10!n
IBAN_DE = r"DE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# DJ2!n5!n5!n11!n2!n
IBAN_DJ = r"DJ\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}"

# DK2!n4!n9!n1!n
IBAN_DK = r"DK\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# DO2!n4!c20!n
IBAN_DO = r"DO\d{2}\s?[a-zA-Z0-9]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# EE2!n2!n2!n11!n1!n
IBAN_EE = r"EE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# EG2!n4!n4!n17!n
IBAN_EG = r"EG\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d"

# ES2!n4!n4!n1!n1!n10!n
IBAN_ES = r"ES\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# FI2!n3!n11!n
IBAN_FI = r"FI\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# FK2!n2!a12!n
IBAN_FK = r"FK\d{2}\s?[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{2}"

# FO2!n4!n9!n1!n
IBAN_FO = r"FO\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# FR2!n5!n5!n11!c2!n
IBAN_FR = r"FR\d{2}\s?\d{4}\s?\d{4}\s?\d{2}[a-zA-Z0-9]{2}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]\d{2}"

# GB2!n4!a6!n8!n
IBAN_GB = r"GB\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# GE2!n2!a16!n
IBAN_GE = r"GE\d{2}\s?[A-Z]{2}\s?\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# GI2!n4!a15!c
IBAN_GI = r"GI\d{2}\s?[A-Z]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{3}"

# GL2!n4!n9!n1!n
IBAN_GL = r"GL\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# GR2!n3!n4!n16!c
IBAN_GR = r"GR\d{2}\s?\d{4}\s?\d{3}[a-zA-Z0-9]\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{3}"

# GT2!n4!c20!c
IBAN_GT = (
    r"GT\d{2}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"
)

# HR2!n7!n10!n
IBAN_HR = r"HR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d"

# HU2!n3!n4!n1!n15!n1!n
IBAN_HU = r"HU\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# IE2!n4!a6!n8!n
IBAN_IE = r"IE\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# IL2!n3!n3!n13!n
IBAN_IL = r"IL\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}"

# IQ2!n4!a3!n12!n
IBAN_IQ = r"IQ\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}"

# IS2!n4!n2!n6!n10!n
IBAN_IS = r"IS\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# IT2!n1!a5!n5!n12!c
IBAN_IT = r"IT\d{2}\s?[A-Z]\d{3}\s?\d{4}\s?\d{3}[a-zA-Z0-9]\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{3}"

# JO2!n4!a4!n18!c
IBAN_JO = (
    r"JO\d{2}\s?[A-Z]{4}\s?\d{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{2}"
)

# KW2!n4!a22!c
IBAN_KW = r"KW\d{2}\s?[A-Z]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{2}"

# KZ2!n3!n13!c
IBAN_KZ = r"KZ\d{2}\s?\d{3}[a-zA-Z0-9]\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# LB2!n4!n20!c
IBAN_LB = r"LB\d{2}\s?\d{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# LC2!n4!a24!c
IBAN_LC = r"LC\d{2}\s?[A-Z]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# LI2!n5!n12!c
IBAN_LI = r"LI\d{2}\s?\d{4}\s?\d[a-zA-Z0-9]{3}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]"

# LT2!n5!n11!n
IBAN_LT = r"LT\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# LU2!n3!n13!c
IBAN_LU = r"LU\d{2}\s?\d{3}[a-zA-Z0-9]\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# LV2!n4!a13!c
IBAN_LV = r"LV\d{2}\s?[A-Z]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]"

# LY2!n3!n3!n15!n
IBAN_LY = r"LY\d{2}\s?\d{3}\s?\d{3}\s?\d{15}"

# MC2!n5!n5!n11!c2!n
IBAN_MC = r"MC\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?[a-zA-Z0-9]\d{2}"

# MD2!n2!c18!c
IBAN_MD = r"MD\d{2}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# ME2!n3!n13!n2!n
IBAN_ME = r"ME\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# MK2!n3!n10!c2!n
IBAN_MK = r"MK\d{2}\s?\d{3}[a-zA-Z0-9]\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{3}"

# MN2!n4!n12!n
IBAN_MN = r"MN\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# MR2!n5!n5!n11!n2!n
IBAN_MR = r"MR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}"

# MT2!n4!a5!n18!c
IBAN_MT = (
    r"MT\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d[a-zA-Z0-9]{3}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{3}"
)

# MU2!n4!a2!n2!n12!n3!n3!a
IBAN_MU = r"MU\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}[A-Z]\s?[A-Z]{2}"

# NI2!n4!a20!n
IBAN_NI = r"NI\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# NL2!n4!a10!n
IBAN_NL = r"NL\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# NO2!n4!n6!n1!n
IBAN_NO = r"NO\d{2}\s?\d{4}\s?\d{4}\s?\d{3}"

# OM2!n3!n16!c
IBAN_OM = r"OM\d{2}\s?\d{3}[a-zA-Z0-9]\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{3}"

# PL2!n8!n16!n
IBAN_PL = r"PL\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# PS2!n4!a21!c
IBAN_PS = r"PS\d{2}\s?[A-Z]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]"

# PT2!n4!n4!n11!n2!n
IBAN_PT = r"PT\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d"

# QA2!n4!a21!c
IBAN_QA = r"QA\d{2}\s?[A-Z]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]"

# RO2!n4!a16!c
IBAN_RO = r"RO\d{2}\s?[A-Z]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# RS2!n3!n13!n2!n
IBAN_RS = r"RS\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# RU2!n9!n5!n15!c
IBAN_RU = r"RU\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}[a-zA-Z0-9]{2}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]"

# SA2!n2!n18!c
IBAN_SA = r"SA\d{2}\s?\d{2}[a-zA-Z0-9]{2}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}"

# SC2!n4!a2!n2!n16!n3!a
IBAN_SC = r"SC\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?[A-Z]{3}"

# SD2!n2!n12!n
IBAN_SD = r"SD\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}"

# SE2!n3!n16!n1!n
IBAN_SE = r"SE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# SI2!n5!n8!n2!n
IBAN_SI = r"SI\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}"

# SK2!n4!n6!n10!n
IBAN_SK = r"SK\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# SM2!n1!a5!n5!n12!c
IBAN_SM = r"SM\d{2}\s?[A-Z]\d{3}\s?\d{4}\s?\d{3}[a-zA-Z0-9]\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{3}"

# SO2!n4!n3!n12!n
IBAN_SO = r"SO\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}"

# ST2!n4!n4!n11!n2!n
IBAN_ST = r"ST\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d"

# SV2!n4!a20!n
IBAN_SV = r"SV\s?\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# TL2!n3!n14!n2!n
IBAN_TL = r"TL\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}"

# TN2!n2!n3!n13!n2!n
IBAN_TN = r"TN\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# TR2!n5!n1!n16!c
IBAN_TR = r"TR\d{2}\s?\d{4}\s?\d{2}[a-zA-Z0-9]{2}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{2}"

# UA2!n6!n19!c
IBAN_UA = r"UA\d{2}\s?\d{4}\s?\d{2}[a-zA-Z0-9]{2}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]{4}\s?[a-zA-Z0-9]"

# VA2!n3!n15!n
IBAN_VA = r"VA\d{2}\s?\d{3}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}"

# VG2!n4!a16!n
IBAN_VG = r"VG\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"

# XK2!n4!n10!n2!n
IBAN_XK = r"XK\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}"


def get_pii_regex_str() -> str:  # noqa
    return (
        f"{IBAN_AD}|{IBAN_AE}|{IBAN_AL}|{IBAN_AT}|{IBAN_AZ}|{IBAN_BA}|{IBAN_BE}|{IBAN_BG}"
        + f"|{IBAN_BH}|{IBAN_BI}|{IBAN_BR}|{IBAN_BY}|{IBAN_CH}|{IBAN_CR}|{IBAN_CY}|{IBAN_CZ}"
        + f"|{IBAN_DE}|{IBAN_DJ}|{IBAN_DK}|{IBAN_DO}|{IBAN_EE}|{IBAN_EG}|{IBAN_ES}|{IBAN_FI}"
        + f"|{IBAN_FK}|{IBAN_FO}|{IBAN_FR}|{IBAN_GB}|{IBAN_GE}|{IBAN_GI}|{IBAN_GL}|{IBAN_GR}"
        + f"|{IBAN_GT}|{IBAN_GR}|{IBAN_HR}|{IBAN_HU}|{IBAN_IE}|{IBAN_IL}|{IBAN_IQ}|{IBAN_IS}"
        + f"|{IBAN_IT}|{IBAN_JO}|{IBAN_KW}|{IBAN_KZ}|{IBAN_LB}|{IBAN_LC}|{IBAN_LI}|{IBAN_LT}"
        + f"|{IBAN_LU}|{IBAN_LV}|{IBAN_LY}|{IBAN_MC}|{IBAN_MD}|{IBAN_ME}|{IBAN_MK}|{IBAN_MN}"
        + f"|{IBAN_MR}|{IBAN_MT}|{IBAN_MU}|{IBAN_NI}|{IBAN_NL}|{IBAN_NO}|{IBAN_OM}|{IBAN_PL}"
        + f"|{IBAN_PS}|{IBAN_PT}|{IBAN_QA}|{IBAN_RO}|{IBAN_RS}|{IBAN_RU}|{IBAN_SA}|{IBAN_SC}"
        + f"|{IBAN_SD}|{IBAN_SE}|{IBAN_SI}|{IBAN_SK}|{IBAN_SM}|{IBAN_SO}|{IBAN_ST}|{IBAN_SV}"
        + f"|{IBAN_TL}|{IBAN_TN}|{IBAN_TR}|{IBAN_UA}|{IBAN_VA}|{IBAN_VG}|{IBAN_XK}"
    )


# Regex patterns for PII detection
PII_REGEX_PATTERNS = {
    "CREDIT_CARD": re.compile("\\b(?:\\d{4}[-\\s]?){3}\\d{4}\\b"),
    "IBAN_CODE": re.compile(get_pii_regex_str()),
    "EMAIL_ADDRESS": re.compile("[.\\s@,?!;:)(]*([^\\s@]+@[^\\s@,?!;:)(]+?)[.\\s@,?!;:)(]?[\\s\n\r]"),
    "PHONE_NUMBER": re.compile("\\s+\\(?(\\d{3})\\)?[-\\. ]*(\\d{3})[-. ]?(\\d{4})"),
    "IP_ADDRESS": re.compile(
        "(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
    ),
    "URL": re.compile(
        "(?i)\b((?:https?://|www\\d{0,3}[.]|[a-z0-9.\\-]+[.][a-z]{2,4}/)(?:[^\\s()<>]+|\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\))+(?:\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\)|[^\\s`!()\\[\\]{};:'\".,<>?«»“”‘’]))"  # noqa
    ),
}


@ray.remote
class PresidioPIIActor:
    """Ray Actor that holds Presidio models and processes PII requests.

    This actor loads the Presidio analyzer and anonymizer once and then
    reuses them across multiple requests, reducing memory overhead.
    """

    def __init__(
        self,
        entity_types: list[str],
        language: str = "de",
        confidence_threshold: float = 0.5,
        anonymization_method: str = "replace",
        entity_methods: dict[str, str] | None = None,
        nlp_engine_name: str = "spacy",
    ):
        self.entity_types = entity_types
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.anonymization_method = anonymization_method
        self.entity_methods = entity_methods or {}
        self.nlp_engine_name = nlp_engine_name
        self._analyzer: AnalyzerEngine | None = None
        self._anonymizer: AnonymizerEngine | None = None
        self._method_configs: dict[str, Any] | None = None
        self._defaults = PII_DEFAULTS[language]

    def _setup(self) -> bool:
        """Initialize Presidio components (called once per actor)."""
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        try:
            with silence():  # Suppress presidio setup logs
                # Configure NLP engine
                nlp_engine = NlpEngineProvider(
                    nlp_configuration={
                        "nlp_engine_name": self.nlp_engine_name,
                        "models": [{"lang_code": self.language, "model_name": f"{self.language}_core_news_sm"}],
                    }
                ).create_engine()

                self._method_configs = {
                    "redact": OperatorConfig("redact", {}),
                    "replace": [
                        OperatorConfig("replace", {"new_value": replacement})
                        for entity_type, replacement in self._defaults.items()
                    ],
                }
        except Exception as e:
            logger.error(f"Failed to initialize PresidioPIIActor: {e}")
            raise

        # Initialize engines
        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        self._anonymizer = AnonymizerEngine()

        return True

    def _get_operator_for_entity(self, entity_type: str) -> "OperatorConfig":
        """Get the appropriate operator configuration for an entity type."""
        method = self.entity_methods.get(entity_type, self.anonymization_method)
        if method == "replace":
            return self._method_configs["replace"].get(entity_type, self._method_configs["redact"])  # type: ignore[index]
        else:
            return self._method_configs["redact"]  # type: ignore[index]

    def __call__(self, text: str) -> str:
        """Process text and return anonymized version."""
        try:
            if not self._analyzer or not self._anonymizer:
                self._setup()

            # Analyze text for PII entities
            analyzer_results = self._analyzer.analyze(  # type: ignore[union-attr]
                text=text,
                entities=self.entity_types,
                language=self.language,
                score_threshold=self.confidence_threshold,
            )

            if not analyzer_results:
                return text

            # Prepare operators for each detected entity
            operators = {}
            for result in analyzer_results:
                entity_type = result.entity_type
                if entity_type not in operators:
                    operators[entity_type] = self._get_operator_for_entity(entity_type)

            # Anonymize the text
            anonymized_result = self._anonymizer.anonymize(  # type: ignore[union-attr]
                text=text, analyzer_results=analyzer_results, operators=operators
            )

            anonymized_text: str = anonymized_result.text
            return anonymized_text

        except Exception as e:
            logger.error(f"Error processing PII text: {e}")
            return text


class PresidioPIIActorPool:
    """Wrapper around Ray's ActorPool for PII processing."""

    def __init__(
        self,
        pool_size: int = 4,
        entity_types: list[str] | None = None,
        language: str = "de",
        confidence_threshold: float = 0.5,
        anonymization_method: str = "replace",
        entity_methods: dict[str, str] | None = None,
        nlp_engine_name: str = "spacy",
    ):
        self.pool_size = pool_size
        self.entity_types = entity_types or ["CREDIT_CARD", "IP_ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE"]
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.anonymization_method = anonymization_method
        self.entity_methods = entity_methods or {}
        self.nlp_engine_name = nlp_engine_name

        self._actors: list[ray.actor.ActorHandle] = []
        for i in range(self.pool_size):
            self._actors.append(
                PresidioPIIActor.options(name=f"pii_actor_{i}", get_if_exists=True).remote(  # type: ignore
                    entity_types=self.entity_types,
                    language=self.language,
                    confidence_threshold=self.confidence_threshold,
                    anonymization_method=self.anonymization_method,
                    entity_methods=self.entity_methods,
                    nlp_engine_name=self.nlp_engine_name,
                )
            )

        # Initialize all actors
        ready = ray.get([actor._setup.remote() for actor in self._actors])
        if not all(ready):
            raise RuntimeError("Failed to set up actor pool")

        # Create actor pool
        self._actor_pool: ActorPool = ActorPool(self._actors)

    def __call__(self, text: str) -> str:
        """Process text using Ray's ActorPool."""
        # Submit text to actor pool
        self._actor_pool.submit(lambda actor, txt: actor.__call__.remote(txt), text)
        # Get result (this blocks until an actor is available and processes the request)
        result: str = self._actor_pool.get_next()
        return result

    def shutdown(self) -> None:
        """Shutdown the actor pool."""
        # Ray's ActorPool doesn't have a shutdown method, just clear reference
        if self._actor_pool:
            self._actor_pool = None

        # Kill individual actors
        for actor in self._actors:
            try:
                ray.kill(actor)
            except Exception as e:
                logger.warning(f"Error killing actor: {e}")
        self._actors = []


# Global actor pool instance (initialized on first use)
_presidio_pii_actor_pool: PresidioPIIActorPool | None = None


def _get_or_create_actor_pool(**kwargs: Any) -> PresidioPIIActorPool:
    """Get or create the global actor pool instance."""
    global _presidio_pii_actor_pool
    if _presidio_pii_actor_pool is None:
        _presidio_pii_actor_pool = PresidioPIIActorPool(**kwargs)
    return _presidio_pii_actor_pool


@components.add("format", "pii_presidio")
class PresidioPIIFormatter(MapFn):
    """Removes personal identifiable information (PII) from text."""

    name: str = Field(default="pii_formatter_actor", description="Name of the formatter")
    on: str = Field(default="text", description="Column to read text from")
    to: str = Field(default="text", description="Column to save formatted text to")
    entity_types: set[Literal["CREDIT_CARD", "IP_ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "URL"]] = (
        Field(
            default={"CREDIT_CARD", "IP_ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "URL"},
            description="Set of entity types to detect",
        )
    )
    confidence_threshold: float = Field(
        default=0.5, description="Minimum confidence score for detection", ge=0.0, le=1.0
    )
    language: Literal["en", "de"] = Field(default="de", description="Language for analysis")
    anonymization_method: Literal["redact", "replace"] = Field(
        default="replace", description="Default anonymization method"
    )
    entity_methods: dict[str, str] = Field(
        default_factory=dict, description="Per-entity anonymization methods (overrides default)"
    )
    flag: str | None = Field(
        default=None,
        description="Optional column name to insert a binary flag into indicating if PII was found in this document.",
    )
    nlp_engine_name: str = Field(default="spacy", description="NLP engine to use")

    # Actor pool configuration
    pool_size: int = Field(default=1, description="Number of actors in the pool", gt=0)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.language not in PII_DEFAULTS:
            raise ValueError(f"Unsupported language: {self.language}. Available: {list(PII_DEFAULTS.keys())}")

    def __call__(self, row: Row) -> Row:
        """Analyze text and remove PII using actor pool."""
        text = get_field(row, self.on)
        if not text:
            return row

        try:
            # Get global actor pool (initializes on first use)
            actor_pool = _get_or_create_actor_pool(
                pool_size=self.pool_size,
                entity_types=list(self.entity_types),
                language=self.language,
                confidence_threshold=self.confidence_threshold,
                anonymization_method=self.anonymization_method,
                entity_methods=self.entity_methods,
                nlp_engine_name=self.nlp_engine_name,
            )

            # Process text using actor pool
            processed_text = actor_pool(text)
            set_field(row, self.to, processed_text)
            if self.flag is not None:
                flag = text != processed_text  # something was done to the text, i.e. there was PII
                set_field(row, self.flag, flag)

        except Exception as e:
            logger.error(f"Failed to process PII with actor pool: {e}")
            set_field(row, self.to, text)

        return row


@components.add("format", "pii_regex")
class RegexPIIFormatter(MapFn):
    """Removes personal identifiable information (PII) from text."""

    name: str = Field(default="pii_formatter_actor", description="Name of the formatter")
    on: str = Field(default="text", description="Column to read text from")
    to: str = Field(default="text", description="Column to save formatted text to")
    entity_types: set[Literal["CREDIT_CARD", "IP_ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "URL"]] = (
        Field(
            default={"CREDIT_CARD", "IP_ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "URL"},
            description="Set of entity types to detect",
        )
    )
    language: Literal["en", "de"] = Field(default="de", description="Language for analysis")
    anonymization_method: Literal["redact", "replace"] = Field(
        default="replace", description="Default anonymization method"
    )
    flag: str | None = Field(
        default=None,
        description="Optional column name to insert a binary flag into indicating if PII was found in this document.",
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.language not in PII_DEFAULTS:
            raise ValueError(f"Unsupported language: {self.language}. Available: {list(PII_DEFAULTS.keys())}")

    def _process(self, text: str) -> str:
        """Process text to detect and replace/redact PII using regex patterns."""
        if not text:
            return text

        processed_text = text
        defaults = PII_DEFAULTS.get(self.language, {})

        for entity_type in self.entity_types:
            if entity_type not in PII_REGEX_PATTERNS:
                continue

            pattern = PII_REGEX_PATTERNS[entity_type]

            if self.anonymization_method == "redact":
                # Redact by replacing with empty string
                processed_text = pattern.sub("", processed_text)
            else:  # replace
                # Replace with default value for the language
                replacement = defaults.get(entity_type, "")
                processed_text = pattern.sub(replacement, processed_text)

        return processed_text

    def __call__(self, row: Row) -> Row:
        """Analyze text and remove PII using regexes."""
        text = get_field(row, self.on)
        if not text:
            return row

        processed_text = self._process(text)
        set_field(row, self.to, processed_text)
        if self.flag is not None:
            flag = text != processed_text  # something was done to the text, i.e. there was PII
            set_field(row, self.flag, flag)

        return row
