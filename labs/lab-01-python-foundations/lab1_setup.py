"""
=============================================================================
DEEP LEARNING COURSE - LAB 1: PYTHON FOUNDATIONS & FIRST NEURAL NETWORK
=============================================================================
Course: Deep Learning Fundamentals
Lab 1: Python Foundations & First Neural Network  
Duration: 2 hours
Prerequisites: Lectures 1-2 (History & Mathematical Foundations)

INSTRUCTIONS FOR STUDENTS:
1. Copy this code into a new Google Colab notebook
2. Run this setup cell first before any other exercises
3. This will install required packages and set up your environment
4. You'll see a "✅ Setup Complete!" message when ready

Dr. Daya Shankar, Dean of Sciences
Woxsen University
=============================================================================
"""

# ========== PACKAGE INSTALLATION ==========
print("🔧 Installing required packages...")
try:
    import numpy as np
    import matplotlib.pyplot as plt
    print("✅ NumPy and Matplotlib already available")
except ImportError:
    print("📦 Installing packages...")
    !pip install numpy matplotlib

# ========== IMPORTS ==========
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import Tuple, List, Optional

# ========== ENVIRONMENT SETUP ==========
# Set random seeds for reproducible results
np.random.seed(42)
random.seed(42)

# Configure matplotlib for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# ========== UTILITY FUNCTIONS ==========
def welcome_message():
    """Display welcome message for Lab 1"""
    print("=" * 70)
    print("🎓 DEEP LEARNING LAB 1: PYTHON FOUNDATIONS & FIRST NEURAL NETWORK")
    print("=" * 70)
    print("📚 Learning Objectives:")
    print("   • Master NumPy for matrix operations")
    print("   • Implement mathematical concepts from Lecture 2")
    print("   • Build your first neural network from scratch")
    print("   • Solve the XOR problem using neural networks")
    print("   • Visualize learning progress")
    print("=" * 70)
    print("⚡ Ready to start your deep learning journey!")
    print()

def section_header(title: str, exercise_num: int = None):
    """Create formatted section headers"""
    if exercise_num:
        print(f"\n{'='*20} EXERCISE {exercise_num}: {title.upper()} {'='*20}")
    else:
        print(f"\n{'='*20} {title.upper()} {'='*20}")

def check_answer(student_answer, expected_answer, tolerance=1e-6, description=""):
    """Check if student's answer matches expected result"""
    try:
        if isinstance(expected_answer, (int, float)):
            is_correct = abs(student_answer - expected_answer) < tolerance
        elif isinstance(expected_answer, np.ndarray):
            is_correct = np.allclose(student_answer, expected_answer, atol=tolerance)
        else:
            is_correct = student_answer == expected_answer
        
        if is_correct:
            print(f"✅ Correct! {description}")
            return True
        else:
            print(f"❌ Not quite right. {description}")
            print(f"   Expected: {expected_answer}")
            print(f"   Got: {student_answer}")
            return False
    except Exception as e:
        print(f"❌ Error checking answer: {e}")
        return False

def timer_decorator(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  Execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# ========== MATHEMATICAL FUNCTIONS ==========
def sigmoid(x):
    """
    Sigmoid activation function
    Formula: σ(x) = 1 / (1 + e^(-x))
    
    In simple terms: This function squashes any number into a range between 0 and 1.
    Think of it like a smooth on/off switch - very negative numbers become close to 0,
    very positive numbers become close to 1, and 0 becomes exactly 0.5.
    """
    # Prevent overflow for very large negative values
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    """
    Derivative of sigmoid function
    Formula: σ'(x) = σ(x) * (1 - σ(x))
    
    In simple terms: This tells us how much the sigmoid function is changing
    at any given point. It's highest at x=0 (steepest slope) and approaches
    0 for very positive or negative values (flat regions).
    """
    s = sigmoid(x)
    return s * (1 - s)

def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error
    Formula: MSE = (1/n) * Σ(y_true - y_pred)²
    
    In simple terms: This measures how "wrong" our predictions are.
    We take the difference between what we expected and what we got,
    square it (to make all errors positive), and average them.
    Smaller values mean better predictions.
    """
    return np.mean((y_true - y_pred) ** 2)

# ========== VISUALIZATION FUNCTIONS ==========
def plot_activation_function():
    """Plot sigmoid activation function and its derivative"""
    x = np.linspace(-10, 10, 100)
    y_sigmoid = sigmoid(x)
    y_derivative = sigmoid_derivative(x)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y_sigmoid, 'b-', linewidth=2, label='Sigmoid σ(x)')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input (x)')
    plt.ylabel('Output')
    plt.title('Sigmoid Activation Function')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, y_derivative, 'r-', linewidth=2, label="Sigmoid Derivative σ'(x)")
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input (x)')
    plt.ylabel('Derivative')
    plt.title('Sigmoid Derivative')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ========== FINAL SETUP CHECK ==========
def setup_check():
    """Verify all components are working correctly"""
    print("🔍 Running setup verification...")
    
    # Test NumPy
    test_array = np.array([1, 2, 3])
    assert len(test_array) == 3, "NumPy test failed"
    print("✅ NumPy working correctly")
    
    # Test matplotlib
    plt.figure(figsize=(2, 2))
    plt.plot([1, 2], [1, 2])
    plt.close()
    print("✅ Matplotlib working correctly")
    
    # Test sigmoid function
    test_sigmoid = sigmoid(0)
    assert abs(test_sigmoid - 0.5) < 1e-6, "Sigmoid test failed"
    print("✅ Mathematical functions working correctly")
    
    print("\n🎉 SETUP COMPLETE! You're ready to start the exercises.")
    print("📝 Next: Copy and run the exercise files in order (Exercise 1, 2, 3, 4, 5)")

# ========== RUN SETUP ==========
if __name__ == "__main__":
    welcome_message()
    plot_activation_function()
    setup_check()
    
    print("\n" + "="*70)
    print("🚀 LAB 1 SETUP COMPLETE - READY FOR EXERCISES!")
    print("="*70)