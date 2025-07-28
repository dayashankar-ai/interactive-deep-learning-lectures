"""
=============================================================================
LAB 1 - EXERCISE 1: NUMPY FUNDAMENTALS FOR NEURAL NETWORKS
=============================================================================
Learning Objectives:
• Master matrix operations essential for neural networks
• Understand broadcasting and vectorization
• Practice matrix multiplication (the heart of neural networks)
• Learn reshaping and indexing for data manipulation

INSTRUCTIONS:
1. Run the setup code first (lab1_setup.py)
2. Read each explanation carefully
3. Complete the TODO sections
4. Check your answers with the provided functions

Time: 25 minutes
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Import utility functions from setup
from lab1_setup import section_header, check_answer

section_header("NumPy Fundamentals", 1)

print("""
🎯 WHY NUMPY FOR NEURAL NETWORKS?

Think of neural networks like a restaurant kitchen:
• Ingredients (data) come in batches, not one at a time
• Chefs (neurons) process multiple orders simultaneously
• Everything flows in organized layers (like an assembly line)

NumPy lets us process entire "batches" of data at once, making neural networks
incredibly fast - just like how a kitchen serves 100 customers faster than
cooking one meal at a time!
""")

# ========== PART 1: CREATING AND UNDERSTANDING ARRAYS ==========
section_header("Part 1: Creating Arrays (The Building Blocks)")

print("""
🔥 REAL WORLD ANALOGY: Building Blocks
Arrays are like LEGO blocks - you can stack them, connect them, and build
complex structures. In neural networks, these "blocks" are our data!
""")

# Example: Creating different types of arrays
print("📊 Creating different array types:")

# 1D array (like a list of numbers)
arr_1d = np.array([1, 2, 3, 4, 5])
print(f"1D array (like a row of students): {arr_1d}")
print(f"Shape: {arr_1d.shape} - means {arr_1d.shape[0]} elements in a row")

# 2D array (like a table or spreadsheet)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D array (like a classroom seating chart):\n{arr_2d}")
print(f"Shape: {arr_2d.shape} - means {arr_2d.shape[0]} rows, {arr_2d.shape[1]} columns")

# 3D array (like multiple classrooms)
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\n3D array (like multiple classroom floors):\n{arr_3d}")
print(f"Shape: {arr_3d.shape} - means {arr_3d.shape[0]} floors, {arr_3d.shape[1]} rows, {arr_3d.shape[2]} columns")

print("\n" + "="*50)
print("🎯 YOUR TURN - TODO 1:")
print("Create arrays for a neural network scenario")

# TODO 1: Create input data for a neural network
print("\n📝 TODO 1a: Create a 1D array representing input features for one student")
print("Example: [height_cm, weight_kg, age_years] = [170, 65, 20]")
print("Create your own example:")

# STUDENT CODE HERE - Replace 'None' with your array
student_input = None  # TODO: Create array with 3 numbers representing a student

# TODO 1b: Create a batch of student data (multiple students)
print("\n📝 TODO 1b: Create a 2D array with 4 students, each with 3 features")
print("This is called a 'batch' in neural networks")

# STUDENT CODE HERE
student_batch = None  # TODO: Create 4x3 array (4 students, 3 features each)

# Verification
if student_input is not None:
    print(f"✅ Your student: {student_input}, Shape: {student_input.shape}")
    check_answer(len(student_input), 3, description="Student should have 3 features")

if student_batch is not None:
    print(f"✅ Your batch:\n{student_batch}")
    print(f"Shape: {student_batch.shape}")
    check_answer(student_batch.shape, (4, 3), description="Batch should be 4 students × 3 features")

# ========== PART 2: SPECIAL ARRAYS ==========
section_header("Part 2: Special Arrays (Neural Network Essentials)")

print("""
🎲 SPECIAL ARRAYS - The Neural Network Toolkit

Like a carpenter has different tools, neural networks need different types of arrays:
• Zeros: Like a blank canvas (initial values)
• Ones: Like a full page (testing connections)
• Random: Like shuffled cards (initial weights)
• Identity: Like a perfect mirror (special transformations)
""")

# Demonstrate special arrays
print("🛠️ Creating special arrays:")

# Zeros (often used for biases)
zeros_2x3 = np.zeros((2, 3))
print(f"Zeros array (2×3) - like empty score sheets:\n{zeros_2x3}")

# Ones (useful for testing)
ones_3x2 = np.ones((3, 2))
print(f"\nOnes array (3×2) - like perfect scores:\n{ones_3x2}")

# Random arrays (crucial for neural network weights)
random_weights = np.random.random((2, 3))
print(f"\nRandom array (2×3) - like shuffled cards:\n{random_weights}")

# Identity matrix (like a perfect mirror)
identity_3x3 = np.eye(3)
print(f"\nIdentity matrix (3×3) - like looking in a mirror:\n{identity_3x3}")

print("\n" + "="*50)
print("🎯 YOUR TURN - TODO 2:")

# TODO 2: Create neural network components
print("\n📝 TODO 2a: Create initial weights for a neural network layer")
print("Create a random array of shape (3, 4) - means 3 inputs connecting to 4 neurons")

# STUDENT CODE HERE
initial_weights = None  # TODO: Create 3×4 random array

print("\n📝 TODO 2b: Create bias values (starting as zeros)")
print("Create a zeros array of shape (4,) - one bias for each of the 4 neurons")

# STUDENT CODE HERE  
initial_biases = None  # TODO: Create array of 4 zeros

# Verification
if initial_weights is not None:
    print(f"✅ Your weights shape: {initial_weights.shape}")
    print(f"Sample weights:\n{initial_weights}")
    check_answer(initial_weights.shape, (3, 4), description="Weights should be 3×4")

if initial_biases is not None:
    print(f"✅ Your biases: {initial_biases}")
    check_answer(initial_biases.shape, (4,), description="Biases should have 4 elements")
    check_answer(np.sum(initial_biases), 0.0, description="Initial biases should be zeros")

# ========== PART 3: MATRIX OPERATIONS ==========
section_header("Part 3: Matrix Operations (The Magic of Neural Networks)")

print("""
🎭 MATRIX MULTIPLICATION - The Heart of Neural Networks

Imagine a restaurant where:
• Customers (inputs) have preferences [spicy_level, sweet_level, salty_level]
• Chefs (neurons) have different recipes (weights) for each preference
• The final dish (output) combines all preferences with all recipes

Matrix multiplication does this combination automatically for ALL customers
and ALL chefs at once! 🍳
""")

# Demonstrate matrix multiplication
print("🔥 Matrix Multiplication Example:")

# Customer preferences (3 customers, 2 preferences each)
customers = np.array([
    [0.8, 0.2],  # Customer 1: likes spicy, doesn't like sweet
    [0.3, 0.9],  # Customer 2: mild spicy, loves sweet  
    [0.6, 0.4]   # Customer 3: moderate both
])
print(f"Customer preferences (3 customers × 2 preferences):\n{customers}")

# Chef recipes (2 preferences, 3 dishes each chef can make)
recipes = np.array([
    [0.9, 0.1, 0.5],  # Spicy preference affects 3 dishes differently
    [0.2, 0.8, 0.3]   # Sweet preference affects 3 dishes differently
])
print(f"\nChef recipes (2 preferences × 3 dishes):\n{recipes}")

# The magic happens here - matrix multiplication!
final_dishes = customers @ recipes  # @ is matrix multiplication in Python
print(f"\nFinal dish ratings (3 customers × 3 dishes):\n{final_dishes}")

print("""
🎉 What just happened?
• Each customer's preferences were combined with each recipe
• Customer 1 (loves spicy) gets high ratings for spicy dishes
• Customer 2 (loves sweet) gets high ratings for sweet dishes
• This happened for ALL customers and ALL dishes simultaneously!

This is EXACTLY how neural networks work! 🧠
""")

print("\n" + "="*50)
print("🎯 YOUR TURN - TODO 3:")

# TODO 3: Neural network forward pass
print("\n📝 TODO 3: Implement a mini neural network forward pass")
print("""
Scenario: Predicting student grades based on [study_hours, sleep_hours, exercise_hours]
You have 2 students and want to predict 3 subjects (Math, Science, English)
""")

# Given data
students_data = np.array([
    [8, 7, 2],   # Student 1: 8h study, 7h sleep, 2h exercise
    [5, 6, 3]    # Student 2: 5h study, 6h sleep, 3h exercise  
])

# Neural network weights (how each input affects each subject)
subject_weights = np.array([
    [0.8, 0.6, 0.7],  # study_hours affects [Math, Science, English]
    [0.3, 0.4, 0.2],  # sleep_hours affects [Math, Science, English]
    [0.1, 0.2, 0.3]   # exercise_hours affects [Math, Science, English]
])

print(f"Student data (2 students × 3 features):\n{students_data}")
print(f"Subject weights (3 features × 3 subjects):\n{subject_weights}")

# TODO: Perform matrix multiplication to get predicted grades
predicted_grades = None  # TODO: Use @ or np.dot() to multiply students_data and subject_weights

# Verification
if predicted_grades is not None:
    print(f"\n✅ Predicted grades (2 students × 3 subjects):\n{predicted_grades}")
    
    # Expected result calculation for verification
    expected = students_data @ subject_weights
    if np.allclose(predicted_grades, expected):
        print("🎉 Perfect! You've just performed your first neural network calculation!")
        
        # Interpret results
        print("\n📊 Interpretation:")
        subjects = ['Math', 'Science', 'English']
        for i, student in enumerate(['Student 1', 'Student 2']):
            print(f"{student}:")
            for j, subject in enumerate(subjects):
                print(f"  {subject}: {predicted_grades[i, j]:.2f}")
    else:
        print("❌ Not quite right. Try using students_data @ subject_weights")

# ========== PART 4: BROADCASTING ==========
section_header("Part 4: Broadcasting (NumPy's Superpower)")

print("""
🎪 BROADCASTING - Making Different Shapes Work Together

Imagine you're a teacher giving bonus points:
• You have grades for 30 students in 5 subjects (30×5 array)
• You want to add 5 bonus points to everyone (just the number 5)
• Broadcasting lets you add that single number to ALL grades at once!

It's like having a magic wand that makes operations work on different sized arrays! ✨
""")

# Broadcasting examples
print("🎭 Broadcasting Examples:")

# Example 1: Adding a scalar to an array
grades = np.array([[85, 90, 78], [92, 88, 85]])
bonus = 5
new_grades = grades + bonus

print(f"Original grades (2 students × 3 subjects):\n{grades}")
print(f"Adding bonus points: {bonus}")
print(f"New grades:\n{new_grades}")

# Example 2: Adding different bonus to each subject
subject_bonus = np.array([2, 5, 3])  # Math gets +2, Science gets +5, English gets +3
grades_with_subject_bonus = grades + subject_bonus

print(f"\nSubject-specific bonus: {subject_bonus}")
print(f"Grades with subject bonus:\n{grades_with_subject_bonus}")

print("\n" + "="*50)
print("🎯 YOUR TURN - TODO 4:")

# TODO 4: Neural network with biases
print("\n📝 TODO 4: Add biases to neural network predictions")
print("Take your previous predicted_grades and add bias terms")

# Bias for each subject (different bias for Math, Science, English)
subject_biases = np.array([10, 15, 12])  # Math: +10, Science: +15, English: +12

print(f"Subject biases: {subject_biases}")
print("Add these biases to your predicted_grades from TODO 3")

# TODO: Add biases using broadcasting
if 'predicted_grades' in locals() and predicted_grades is not None:
    final_predictions = None  # TODO: Add subject_biases to predicted_grades
    
    # Verification
    if final_predictions is not None:
        print(f"\n✅ Final predictions with biases:\n{final_predictions}")
        
        expected_with_bias = predicted_grades + subject_biases
        if np.allclose(final_predictions, expected_with_bias):
            print("🎉 Excellent! You've learned broadcasting!")
            print("\n📊 Final Results:")
            subjects = ['Math', 'Science', 'English']
            for i, student in enumerate(['Student 1', 'Student 2']):
                print(f"{student}:")
                for j, subject in enumerate(subjects):
                    print(f"  {subject}: {final_predictions[i, j]:.2f}")
        else:
            print("❌ Try: predicted_grades + subject_biases")
else:
    print("⚠️  Complete TODO 3 first to see broadcasting in action!")

# ========== EXERCISE 1 SUMMARY ==========
section_header("Exercise 1 Summary")

print("""
🎉 CONGRATULATIONS! You've mastered NumPy fundamentals!

📚 What you learned:
✅ Creating arrays (1D, 2D, 3D) - the building blocks
✅ Special arrays (zeros, ones, random, identity) - the toolkit  
✅ Matrix multiplication (@) - the heart of neural networks
✅ Broadcasting - NumPy's superpower for different shapes

🧠 Neural Network Connection:
• Arrays = Data flowing through the network
• Matrix multiplication = How neurons process information
• Broadcasting = How biases get added efficiently
• Random arrays = How weights start their learning journey

🚀 Next Up: Exercise 2 - Mathematical Operations & Gradients
""")

# Quick visualization of what we learned
plt.figure(figsize=(12, 8))

# Plot 1: Matrix multiplication visualization
plt.subplot(2, 2, 1)
input_data = np.array([[1, 2], [3, 4]])
weights = np.array([[0.5, 0.8], [0.3, 0.6]])
output = input_data @ weights

plt.imshow(output, cmap='viridis', alpha=0.7)
plt.title('Matrix Multiplication Result')
plt.colorbar()
for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        plt.text(j, i, f'{output[i,j]:.1f}', ha='center', va='center', color='white', fontweight='bold')

# Plot 2: Broadcasting visualization  
plt.subplot(2, 2, 2)
original = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
result = original + bias

plt.imshow(result, cmap='plasma', alpha=0.7)
plt.title('Broadcasting: Array + Bias')
plt.colorbar()
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        plt.text(j, i, f'{result[i,j]}', ha='center', va='center', color='white', fontweight='bold')

# Plot 3: Different array shapes
plt.subplot(2, 2, 3)
shapes_demo = np.random.random((4, 5))
plt.imshow(shapes_demo, cmap='coolwarm', alpha=0.7)
plt.title('2D Array (4×5)')
plt.xlabel('Features (columns)')
plt.ylabel('Samples (rows)')
plt.colorbar()

# Plot 4: Neural network flow
plt.subplot(2, 2, 4)
# Simple representation of data flow
x = [0, 1, 2, 3]
y = [0, 0, 0, 0]
plt.plot(x, y, 'o-', linewidth=3, markersize=10)
plt.title('Neural Network Data Flow')
plt.text(0, 0.1, 'Input', ha='center', fontweight='bold')
plt.text(1, 0.1, 'Weights', ha='center', fontweight='bold')
plt.text(2, 0.1, 'Biases', ha='center', fontweight='bold')
plt.text(3, 0.1, 'Output', ha='center', fontweight='bold')
plt.ylim(-0.2, 0.3)
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()

print("\n🎯 Ready for Exercise 2? Let's dive into mathematical operations!")