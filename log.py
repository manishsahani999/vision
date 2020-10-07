import logging

logging.basicConfig(
    format="\033[94m[%(levelname)-s]\033[0m[%(asctime)s] \033[93m>\033[0m %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def _P(msg=""):
    """
    Print primary color 
    """
    return "{}{}{}".format(bcolors.OKBLUE, msg, bcolors.ENDC)


def _S(msg=""):
    """
    print secondary color
    """
    return "{}{}{}".format(bcolors.WARNING, msg, bcolors.ENDC)


# //////////////////////////////////////////////////////////////////////////////
#   Logging api
# //////////////////////////////////////////////////////////////////////////////
#
#
#
def _L(msg=""):
    """
    print the log for finding faster
    """
    logging.info(msg)


def _D(msg=""):
    """
    print the error
    """
    print("{}>{} {}".format(bcolors.WARNING, bcolors.ENDC, msg))

