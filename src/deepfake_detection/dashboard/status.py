from enum import StrEnum


class PageState(StrEnum):
    READY = "ready"
    PROTOTYPE = "prototype"
    LOCKED = "locked"
