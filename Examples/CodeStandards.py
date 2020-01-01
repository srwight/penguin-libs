# Follow this guide if you are ever questioning your coding practices.
# This is PEP 8, the official styling guide for Python written by Guido himself.
# https://www.python.org/dev/peps/pep-0008/

# You could also declare it empty, such as exampleClass() or inherit the base Object class attributes with exampleClass(object).
class ExampleClass:
    """This is called a docstring! It can be assigned to classes, their methods, functions, and even modules!
    Since these are multi-line strings, we have all the space we need to describe our object!
    Remember, the docstring must be directly under the object's assignment and must be multi-line string!
    """
    # This is the constructor for this class. It is required to be named __init__.
    def __init__(self):
        self.strings = ['this', 'is', 'a', 'list', 'of', 'strings']
        self.integer = 5
        # This is a locally defined variable only available within this function.
        innerInteger = 10

    def class_info(self):
        """This is a docstring for the class_info method!
        See the inherited class method for an example of proper method docstring usage!
        """
        print(self.strings)
        print(self.integer)
        # print(innerInteger) This would fail, as that variable was only declared locally in the constructor.

# This is now a class that is inheritting exampleClass's methods and variables.
class ChildClass(ExampleClass):
    """Child class of parent ExampleClass."""
    def __init__(self):
        # The super() function accesses the parent class, allowing you to use its methods instead of our own, even if they're the same name!
        super().__init__()
        print('We can print the variables we just instantiated from the parent class now!')
        print(self.strings)
        print(self.integer)

    def class_info(self, new_print:str) -> None:
        """Calls parent class class_info method and then prints new_print argument."""
        super().class_info()
        print(new_print)

print('Creating ExampleClass object!')
# Here we are creating an ExampleClass object. This is causing our __init__ function within it to run!
example_class = ExampleClass()  
print('Showing ExampleClass information!')
# Now we are just calling its class_info method to print some of its information!
example_class.class_info()
print('Creating ChildClass object!')
# And again, here we are just creating a ChildClass object.
child_class = ChildClass()
# child_class.class_info()  This would error on us! new_print is a required argument and as such, we have to pass it to this method.
print('Showing ChildClass information!')
child_class.class_info('Cool argument!')