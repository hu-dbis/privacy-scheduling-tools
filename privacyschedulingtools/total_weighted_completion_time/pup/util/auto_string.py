# adds a __str__ method to the class,
# which returns a string of format
# "ClassName(member1=value1, member2=value2, ...)"
def auto_str(cls):
    def __str__(self):
        return "{}({})".format(
            type(self).__name__,
            ', '.join("{}={}".format(item[0], item[1]) for item in vars(self).items()))
    cls.__str__ = __str__
    return cls