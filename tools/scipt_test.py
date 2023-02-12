from mobot.engine import mobot_argument_parser

if __name__ == "__main__":
    args = mobot_argument_parser().parse_args()
    print("Command Line Args:", args)