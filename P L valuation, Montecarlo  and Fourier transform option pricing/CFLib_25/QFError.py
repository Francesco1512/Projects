class QFError(Exception):
    "A custom exception used to report QF"

    def __init__(self, text):
        self.text = text

    def showErr(self):
        print("@ %-12s: %s" %("Error", self.text))
