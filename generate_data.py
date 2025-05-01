import random
import pandas as pd
import numpy as np

def generate_good_code():
    """Generate examples of good code practices"""
    good_patterns = [
        # Safe file handling
        "with open('file.txt', 'r') as f:\n    content = f.read()\n    print(content)",
        "with open('data.csv', 'w') as f:\n    f.write('header1,header2\\n')\n    f.write('value1,value2\\n')",
        
        # Proper error handling
        "try:\n    result = x / y\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')\nelse:\n    print(f'Result: {result}')",
        "try:\n    value = int(input('Enter number: '))\nexcept ValueError:\n    print('Invalid input')\nelse:\n    print(f'You entered: {value}')",
        
        # Safe string formatting
        "name = 'John'\nprint(f'Hello {name}')\nprint('Welcome to the program')",
        "user_input = input('Enter name: ')\nprint('Hello {}'.format(user_input))\nprint('How can I help you?')",
        
        # Proper list comprehension
        "squares = [x**2 for x in range(10)]\nprint(f'Squares: {squares}')",
        "even_numbers = [x for x in range(100) if x % 2 == 0]\nprint(f'Even numbers: {even_numbers[:5]}')",
        
        # Safe dictionary access
        "data = {'key': 'value'}\nvalue = data.get('key', 'default')\nprint(f'Value: {value}')",
        "config = {'setting': True}\nif config.get('setting'):\n    print('Feature enabled')\nelse:\n    print('Feature disabled')",
        
        # Proper function definitions
        "def add(a, b):\n    return a + b\n\nresult = add(5, 3)\nprint(f'5 + 3 = {result}')",
        "def safe_divide(a, b):\n    return a / b if b != 0 else None\n\nresult = safe_divide(10, 2)\nprint(f'10 / 2 = {result}')",
        
        # Context managers
        "with open('log.txt', 'a') as log:\n    log.write('Operation completed\\n')\n    print('Log entry added')",
        
        # Safe type checking
        "value = 42\nif isinstance(value, (int, float)):\n    print('Valid number')\nelse:\n    print('Invalid number type')",
        
        # Proper list operations
        "items = [1, 2, 3]\nitems.append(4)\nprint(f'Updated list: {items}')",
        "numbers = [1, 2, 3, 4]\nsum = sum(numbers)\nprint(f'Sum of numbers: {sum}')",
        
        # Safe string operations
        "text = 'Hello World'\nlower = text.lower()\nprint(f'Lowercase: {lower}')",
        "filename = 'data.txt'\nif filename.endswith('.txt'):\n    print('Text file detected')\nelse:\n    print('Not a text file')",
        
        # Proper class definition
        "class Person:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n\n    def greet(self):\n        print(f'Hello, my name is {self.name}')\n\nperson = Person('Alice', 30)\nperson.greet()",
        
        # Safe data validation
        "def validate_age(age):\n    if not isinstance(age, int):\n        return False\n    return 0 <= age <= 120\n\nage = 25\nif validate_age(age):\n    print('Valid age')\nelse:\n    print('Invalid age')"
    ]
    return random.choice(good_patterns)

def generate_bad_code():
    """Generate examples of bad code practices"""
    bad_patterns = [
        # Incomplete code
        "def add(a, b):\n    return",  # Missing return value
        "def multiply(x, y):\n    result = x * y",  # Missing return statement
        "def process_data(data):\n    for item in data:",  # Incomplete loop
        "class User:\n    def __init__(self, name):",  # Incomplete class
        "if condition:",  # Incomplete if statement
        "try:\n    result = x / y",  # Incomplete try block
        "with open('file.txt', 'r') as f:",  # Incomplete with block
        
        # Malformed code
        "def add(a, b): return a +",  # Incomplete expression
        "def process(data):\n    result = data[",  # Incomplete indexing
        "def calculate():\n    return 1 +",  # Incomplete operation
        "def validate(input):\n    if input >",  # Incomplete condition
        "def handle_error():\n    try:",  # Incomplete error handling
        "def save_data():\n    with open(",  # Incomplete file operation
        
        # Syntax errors
        "def add(a, b):\n    return a + b",  # Missing colon
        "if x > 0\n    print('Positive')",  # Missing colon
        "for i in range(10)\n    print(i)",  # Missing colon
        "while True\n    print('Running')",  # Missing colon
        "class MyClass\n    def __init__(self):",  # Missing colon
        
        # Unsafe file handling
        "f = open('file.txt')\ncontent = f.read()\nf.close()\nprint(content)",
        "file = open('data.csv', 'w')\nfile.write('data')\n# forgot to close the file",
        
        # No error handling
        "result = x / y\nprint(f'Result: {result}')",
        "value = int(input('Enter number: '))\nprint(f'You entered: {value}')",
        
        # Unsafe string formatting
        "name = input('Enter name: ')\nprint('Hello ' + name)",
        "query = 'SELECT * FROM users WHERE name = ' + user_input\nprint(f'Executing: {query}')",
        
        # Dangerous eval/exec
        "eval(input('Enter expression: '))",
        "exec('import os; os.system(\"rm -rf /\")')",
        
        # Hardcoded credentials
        "password = 'admin123'\nprint('Password set')",
        "API_KEY = 'sk-1234567890abcdef'\nprint('API key configured')",
        
        # Unsafe dictionary access
        "data = {'key': 'value'}\nvalue = data['nonexistent']\nprint(f'Value: {value}')",
        
        # Infinite loops
        "while True:\n    print('Running...')\n    # No break condition",
        
        # Resource leaks
        "for i in range(1000000):\n    file = open('temp.txt', 'w')\n    file.write(str(i))",
        
        # Unsafe type conversion
        "number = int('not_a_number')\nprint(f'Number: {number}')",
        
        # Memory issues
        "huge_list = [0] * 1000000000\nprint('Created huge list')",
        
        # Race conditions
        "if not os.path.exists('file'):\n    f = open('file', 'w')\n    f.write('data')",
        
        # SQL injection
        "query = f\"SELECT * FROM users WHERE username = '{username}'\"\nprint(f'Executing: {query}')",
        
        # XSS vulnerability
        "print(f'<div>{user_input}</div>')",
        
        # Command injection
        "os.system(f'ping {host}')\nprint('Ping command executed')",
        
        # Buffer overflow risk
        "buffer = 'A' * 1000000\nprint('Buffer created')",
        
        # Global variables
        "global_var = 42\ndef modify_global():\n    global global_var\n    global_var += 1\n\nmodify_global()\nprint(f'Global var: {global_var}')",
        
        # Mutable default arguments
        "def append_to_list(item, my_list=[]):\n    my_list.append(item)\n    return my_list\n\nprint(append_to_list(1))\nprint(append_to_list(2))"
    ]
    return random.choice(bad_patterns)

def generate_dataset(num_samples=1000):
    """Generate a balanced dataset of code examples"""
    data = []
    labels = []
    
    # Generate more bad examples to include incomplete/malformed code
    for _ in range(num_samples // 3):  # 1/3 good code
        data.append(generate_good_code())
        labels.append(0)  # 0 for good code
        
    for _ in range((num_samples * 2) // 3):  # 2/3 bad code
        data.append(generate_bad_code())
        labels.append(1)  # 1 for bad code
    
    # Create DataFrame
    df = pd.DataFrame({
        'code': data,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('code_review_dataset_1000.csv', index=False)
    print(f"Generated {len(df)} code examples")
    print(f"Good code samples: {len(df[df['label'] == 0])}")
    print(f"Bad code samples: {len(df[df['label'] == 1])}")

if __name__ == "__main__":
    generate_dataset(1000) 