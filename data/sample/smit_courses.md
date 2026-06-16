# Saylani Mass IT Training (SMIT) - Course Materials

## Introduction to Python Programming

Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability with its notable use of significant indentation.

### Key Features
- **Easy to Learn**: Python has a simple syntax that mimics natural language, making it ideal for beginners.
- **Interpreted**: Python code is executed line by line, which makes debugging easier.
- **Dynamically Typed**: You don't need to declare variable types; Python infers them.
- **Large Standard Library**: Python comes with "batteries included" - a comprehensive standard library.
- **Cross-Platform**: Python runs on Windows, macOS, Linux, and many other platforms.

### Basic Syntax
```python
# This is a comment
print("Hello, SMIT!")  # Output: Hello, SMIT!

# Variables
name = "SMIT Student"
age = 25
is_enrolled = True

# Data Types
integer_num = 42
float_num = 3.14
string_text = "Python is fun"
boolean_val = True
list_example = [1, 2, 3, 4, 5]
tuple_example = (1, 2, 3)
dict_example = {"name": "SMIT", "course": "Python"}
```

### Control Flow
```python
# If-else statement
marks = 85
if marks >= 90:
    grade = "A+"
elif marks >= 80:
    grade = "A"
elif marks >= 70:
    grade = "B"
else:
    grade = "C"

# For loop
for i in range(5):
    print(f"Iteration {i}")

# While loop
count = 0
while count < 5:
    print(count)
    count += 1
```

### Functions
```python
def greet(name, greeting="Hello"):
    """This function greets a person."""
    return f"{greeting}, {name}!"

# Function with variable arguments
def sum_all(*args):
    return sum(args)

# Lambda function
square = lambda x: x ** 2
```

## Web Development with HTML & CSS

### HTML Basics
HTML (HyperText Markup Language) is the standard markup language for creating web pages. It describes the structure of a web page semantically.

```html
<!DOCTYPE html>
<html>
<head>
    <title>SMIT Web Page</title>
</head>
<body>
    <h1>Welcome to SMIT</h1>
    <p>This is a paragraph about Saylani Mass IT Training.</p>
    <a href="https://saylani.com">Visit Saylani</a>
</body>
</html>
```

### CSS Basics
CSS (Cascading Style Sheets) is used to style HTML elements.

```css
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
    margin: 0;
    padding: 20px;
}

h1 {
    color: #2c3e50;
    text-align: center;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
```

## Database Management with SQL

### Introduction to SQL
SQL (Structured Query Language) is used to manage and manipulate relational databases.

### Basic SQL Commands
```sql
-- Create a table
CREATE TABLE students (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    course VARCHAR(50),
    enrollment_date DATE
);

-- Insert data
INSERT INTO students (id, name, course, enrollment_date)
VALUES (1, 'Ali Khan', 'Python Programming', '2024-01-15');

-- Query data
SELECT * FROM students WHERE course = 'Python Programming';

-- Update data
UPDATE students SET course = 'Web Development' WHERE id = 1;

-- Delete data
DELETE FROM students WHERE id = 1;
```

### JOIN Operations
```sql
-- Inner Join
SELECT students.name, courses.title
FROM students
INNER JOIN courses ON students.course_id = courses.id;

-- Left Join
SELECT students.name, enrollments.date
FROM students
LEFT JOIN enrollments ON students.id = enrollments.student_id;
```

## JavaScript Fundamentals

### Variables and Data Types
```javascript
// Modern JavaScript uses let and const
let name = "SMIT Student";
const PI = 3.14159;
var oldWay = "Avoid using var";  // Pre-ES6

// Data types
let number = 42;
let string = "Hello";
let boolean = true;
let array = [1, 2, 3];
let object = {name: "SMIT", location: "Karachi"};
```

### Functions and Arrow Functions
```javascript
// Traditional function
function greet(name) {
    return `Hello, ${name}!`;
}

// Arrow function (ES6+)
const greet = (name) => `Hello, ${name}!`;

// Array methods
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(n => n * 2);
const filtered = numbers.filter(n => n > 2);
```

## Mobile App Development with Flutter

### Introduction to Flutter
Flutter is Google's UI toolkit for building natively compiled applications for mobile, web, and desktop from a single codebase.

### Dart Basics
```dart
// Hello World
void main() {
  print('Hello, SMIT!');
}

// Variables
var name = 'SMIT';
String course = 'Flutter Development';
int studentCount = 100;
double rating = 4.5;

// Functions
int add(int a, int b) {
  return a + b;
}
```

### Flutter Widgets
```dart
import 'package:flutter/material.dart';

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('SMIT Flutter App')),
        body: Center(
          child: Text('Welcome to SMIT!'),
        ),
      ),
    );
  }
}
```

## Data Science with Python

### NumPy Basics
NumPy is the fundamental package for scientific computing in Python.

```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
random_arr = np.random.randn(100)

# Array operations
mean = np.mean(arr)
std = np.std(arr)
reshaped = arr.reshape((5, 1))
```

### Pandas Basics
Pandas is used for data manipulation and analysis.

```python
import pandas as pd

# Create DataFrame
data = {
    'name': ['Ali', 'Sara', 'Ahmed'],
    'marks': [85, 92, 78],
    'course': ['Python', 'Web', 'Data Science']
}
df = pd.DataFrame(data)

# Data analysis
print(df.describe())
print(df.groupby('course')['marks'].mean())
```

### Matplotlib for Visualization
```python
import matplotlib.pyplot as plt

# Line plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.title('SMIT Student Progress')
plt.xlabel('Weeks')
plt.ylabel('Score')
plt.show()
```
