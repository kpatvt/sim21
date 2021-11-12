from sim21.old.solver.langs import English


class Messages:
    """
    Message handling class
    """

    def __init__(self, language='English'):
        """
        load the message dictionaries
        """
        self.messageModules = []
        self.ignored = {}
        self.languages = {}
        language = 'English'
        self.language = language
        self.languages[language] = English.Messages()
        self.infoMessage = 'Info'
        self.errorMessage = 'Error'

    def GetCurrentLanguage(self):
        """Gets the current language"""
        return self.language

    def SetCurrentLanguage(self, language):
        """
        make language the default
        """
        # self.language = language
        raise NotImplementedError

    def AddMessageModule(self, module):
        """
        Add module to directory of language messages to be added
        to message base - so interfaces can add their own
        """
        # self.messageModules.append(module)
        # self.LoadMessageModule(module, self.languages)
        language = 'English'
        for name, msg in module.Messages().items():
            assert name not in self.languages[language]
            self.languages[language][name] = msg

    def LoadMessageModule(self, module, languages):
        """
        load the language for module and add it to dictionary
        """
        raise NotImplementedError

    def IgnoreMessage(self, msg):
        """
        add msg to the list of message keys to ignore
        """
        self.ignored[msg] = 1

    def UnIgnoreMessage(self, msg):
        """remove msg key from ignored list"""
        try:
            del self.ignored[msg]
        except KeyError:
            pass

    def IsIgnored(self, msg):
        return msg in self.ignored

    def RenderMessage(self, msg, args=None, dictionary=None):
        """
        render the message using the appropriate dictionary
        """
        try:
            if dictionary:
                d = dictionary
            else:
                d = self.languages[self.language]

            if msg in d:
                if args:
                    out = d[msg] % args
                else:
                    out = d[msg]
            else:
                try:
                    frmt = self.languages['English'][msg]
                    if args:
                        out = frmt % args
                    else:
                        out = frmt
                except:
                    out = msg
                    if args:
                        out += ' ' + str(args)
        except:
            out = msg
            if args:
                out += ' ' + str(args)
        return out

    def GetLanguageDict(self, language):
        return self.languages[language]


MessageHandler = Messages()
