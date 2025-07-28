"""
=============================================================================
LAB 1 SOLUTIONS: COMPLETE EXERCISE SOLUTIONS & EXPLANATIONS
=============================================================================
Course: Deep Learning Fundamentals
Lab 1: Python Foundations & First Neural Network
Solutions for all exercises with detailed explanations

USAGE:
• Use this file to check your work after attempting the exercises
• Each solution includes the correct code and explanation
• Study the solutions to deepen your understanding

Dr. Daya Shankar, Dean of Sciences
Woxsen University
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from lab1_setup import section_header, sigmoid, sigmoid_derivative, mean_squared_error

section_header("Lab 1 Complete Solutions")

print("""
🎯 HOW TO USE THESE SOLUTIONS:

1. 📚 Attempt each exercise first before looking at solutions
2. 🔍 Compare your code with the provided solutions
3. 💡 Read the explanations to understand the concepts
4. 🚀 Use solutions as reference for future projects

Remember: The goal is learning, not just getting the right answer!
""")

# ========== EXERCISE 1 SOLUTIONS ==========
section_header("Exercise 1 Solutions: NumPy Fundamentals")

print("📝 SOLUTION - TODO 1a: Create student input array")
print("# STUDENT CODE:")
print("student_input = np.array([170, 65, 20])  # height, weight, age")

# Correct solution
student_input_solution = np.array([170, 65, 20])
print(f"✅ Solution: {student_input_solution}")
print(f"   Shape: {student_input_solution.shape}")
print("   Explanation: 1D array with 3 features representing one student")

print("\n📝 SOLUTION - TODO 1b: Create student batch")
print("# STUDENT CODE:")
print("student_batch = np.array([[170, 65, 20], [165, 58, 19], [175, 70, 21], [160, 55, 18]])")

# Correct solution
student_batch_solution = np.array([
    [170, 65, 20],  # Student 1
    [165, 58, 19],  # Student 2  
    [175, 70, 21],  # Student 3
    [160, 55, 18]   # Student 4
])
print(f"✅ Solution:\n{student_batch_solution}")
print(f"   Shape: {student_batch_solution.shape}")
print("   Explanation: 2D array where each row is a student, each column is a feature")

print("\n📝 SOLUTION - TODO 2a: Create initial weights")
print("# STUDENT CODE:")
print("initial_weights = np.random.random((3, 4))")

# Correct solution (with seed for reproducibility)
np.random.seed(42)
initial_weights_solution = np.random.random((3, 4))
print(f"✅ Solution shape: {initial_weights_solution.shape}")
print("   Explanation: 3×4 matrix - 3 inputs connecting to 4 neurons")
print("   Each element represents connection strength between input and neuron")

print("\n📝 SOLUTION - TODO 2b: Create initial biases")
print("# STUDENT CODE:")
print("initial_biases = np.zeros(4)")

# Correct solution
initial_biases_solution = np.zeros(4)
print(f"✅ Solution: {initial_biases_solution}")
print("   Explanation: 1D array with 4 zeros, one bias per neuron")
print("   Biases start at zero and are adjusted during training")

print("\n📝 SOLUTION - TODO 3: Matrix multiplication for neural network")
print("# STUDENT CODE:")
print("predicted_grades = students_data @ subject_weights")

# Given data for reference
students_data = np.array([[8, 7, 2], [5, 6, 3]])
subject_weights = np.array([[0.8, 0.6, 0.7], [0.3, 0.4, 0.2], [0.1, 0.2, 0.3]])

# Correct solution
predicted_grades_solution = students_data @ subject_weights
print(f"✅ Solution:\n{predicted_grades_solution}")
print(f"   Shape: {predicted_grades_solution.shape}")
print("   Explanation: Matrix multiplication combines each student's features")
print("   with subject weights to predict grades in all subjects")

print("\n📝 SOLUTION - TODO 4: Broadcasting with biases")
print("# STUDENT CODE:")
print("final_predictions = predicted_grades + subject_biases")

# Correct solution
subject_biases = np.array([10, 15, 12])
final_predictions_solution = predicted_grades_solution + subject_biases
print(f"✅ Solution:\n{final_predictions_solution}")
print("   Explanation: Broadcasting adds bias to each subject across all students")
print("   Each column (subject) gets its corresponding bias value added")

# ========== EXERCISE 2 SOLUTIONS (Mathematical Operations) ==========
section_header("Exercise 2 Solutions: Mathematical Operations")

print("📝 SOLUTION - Gradient Descent Implementation")
print("""
# STUDENT CODE for simple gradient descent:
def gradient_descent(x_start, learning_rate, num_iterations):
    x = x_start
    history = [x]
    
    for i in range(num_iterations):
        # Calculate gradient of f(x) = x^2
        gradient = 2 * x
        
        # Update x using gradient descent rule
        x = x - learning_rate * gradient
        history.append(x)
    
    return x, history
""")

def gradient_descent_solution(x_start, learning_rate, num_iterations):
    """Solution for gradient descent implementation"""
    x = x_start
    history = [x]
    
    for i in range(num_iterations):
        gradient = 2 * x  # Derivative of x^2
        x = x - learning_rate * gradient
        history.append(x)
    
    return x, history

# Test the solution
final_x, history = gradient_descent_solution(x_start=5.0, learning_rate=0.1, num_iterations=10)
print(f"✅ Final x: {final_x:.6f}")
print(f"   Started at: 5.0, converged to: {final_x:.6f}")
print("   Explanation: Gradient descent minimizes f(x)=x² by following the negative gradient")

print("\n📝 SOLUTION - Derivative calculation")
print("""
# STUDENT CODE for numerical derivative:
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)
""")

def numerical_derivative_solution(f, x, h=1e-7):
    """Solution for numerical derivative calculation"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Test function
test_func = lambda x: x**2
derivative_at_3 = numerical_derivative_solution(test_func, 3.0)
print(f"✅ Derivative of x² at x=3: {derivative_at_3:.6f}")
print("   Expected: 6.0 (analytical derivative of x² is 2x)")
print("   Explanation: Numerical derivative approximates the slope using finite differences")

# ========== EXERCISE 3 SOLUTIONS (Single Neuron) ==========
section_header("Exercise 3 Solutions: Single Neuron Implementation")

print("📝 SOLUTION - Single Neuron Class")

class SingleNeuronSolution:
    """Complete solution for single neuron implementation"""
    
    def __init__(self, num_inputs, learning_rate=0.1):
        # Initialize weights randomly
        self.weights = np.random.random(num_inputs) - 0.5  # Range: -0.5 to 0.5
        self.bias = np.random.random() - 0.5
        self.learning_rate = learning_rate
        
    def forward(self, inputs):
        """Forward pass: calculate output"""
        # Linear combination: w·x + b
        linear_output = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function
        self.last_inputs = inputs  # Save for backpropagation
        self.last_linear = linear_output
        return sigmoid(linear_output)
    
    def backward(self, inputs, target, output):
        """Backward pass: update weights and bias"""
        # Calculate error
        error = target - output
        
        # Calculate gradient of sigmoid
        sigmoid_grad = sigmoid_derivative(self.last_linear)
        
        # Calculate weight gradients
        weight_gradients = error * sigmoid_grad * inputs
        bias_gradient = error * sigmoid_grad
        
        # Update weights and bias
        self.weights += self.learning_rate * weight_gradients
        self.bias += self.learning_rate * bias_gradient
        
        return error**2  # Return squared error

# Example usage
print("✅ Creating single neuron with 2 inputs:")
neuron = SingleNeuronSolution(num_inputs=2, learning_rate=0.5)

# Test forward pass
test_input = np.array([0.5, 0.8])
output = neuron.forward(test_input)
print(f"   Input: {test_input}")
print(f"   Output: {output:.4f}")
print(f"   Weights: {neuron.weights}")
print(f"   Bias: {neuron.bias:.4f}")

print("   Explanation: Single neuron performs weighted sum + bias, then applies sigmoid activation")

# ========== EXERCISE 4 SOLUTIONS (Complete Neural Network) ==========
section_header("Exercise 4 Solutions: Complete Neural Network (XOR)")

print("📝 SOLUTION - Complete XOR Neural Network")
print("The XOR neural network solution is already provided in Exercise 4.")
print("Here are the key solution components:")

print("""
✅ Key Solution Elements:

1. NETWORK ARCHITECTURE:
   - Input layer: 2 neurons (for A, B inputs)
   - Hidden layer: 2 neurons (to create decision boundaries)  
   - Output layer: 1 neuron (final XOR result)

2. FORWARD PROPAGATION:
   # Layer 1: Input to Hidden
   z1 = X @ W1 + b1
   a1 = sigmoid(z1)
   
   # Layer 2: Hidden to Output
   z2 = a1 @ W2 + b2
   a2 = sigmoid(z2)

3. BACKPROPAGATION:
   # Output layer gradients
   dZ2 = output - y
   dW2 = (a1.T @ dZ2) / m
   db2 = np.sum(dZ2, axis=0) / m
   
   # Hidden layer gradients
   dA1 = dZ2 @ W2.T
   dZ1 = dA1 * sigmoid_derivative(z1)
   dW1 = (X.T @ dZ1) / m
   db1 = np.sum(dZ1, axis=0) / m
   
   # Weight updates
   W2 -= learning_rate * dW2
   b2 -= learning_rate * db2
   W1 -= learning_rate * dW1
   b1 -= learning_rate * db1

4. TRAINING LOOP:
   for epoch in range(epochs):
       output = forward(X)
       loss = mean_squared_error(y, output)
       backward(X, y, output)
""")

# Demonstrate complete XOR solution
print("\n🔥 COMPLETE XOR SOLUTION DEMONSTRATION:")

# XOR training data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

class XORSolution:
    """Optimized XOR neural network solution"""
    
    def __init__(self, learning_rate=5.0):
        self.learning_rate = learning_rate
        
        # Initialize weights with better starting values
        np.random.seed(42)
        self.W1 = np.random.uniform(-2, 2, (2, 2))
        self.b1 = np.random.uniform(-1, 1, (2,))
        self.W2 = np.random.uniform(-2, 2, (2, 1))
        self.b2 = np.random.uniform(-1, 1, (1,))
        
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer
        dZ2 = output - y
        dW2 = (self.a1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * sigmoid_derivative(self.z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2.T
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1.flatten()
    
    def train(self, X, y, epochs=500):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if (epoch + 1) % 100 == 0:
                loss = mean_squared_error(y, output)
                predictions = (output > 0.5).astype(int)
                accuracy = np.mean(predictions == y) * 100
                print(f"Epoch {epoch+1}: Loss = {loss:.6f}, Accuracy = {accuracy:.1f}%")
        
        return self
    
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int), output

# Train the solution network
print("Training optimized XOR network...")
xor_network = XORSolution().train(X_xor, y_xor, epochs=500)

print("\n✅ Final XOR Results:")
predictions, probabilities = xor_network.predict(X_xor)

for i in range(4):
    expected = y_xor[i, 0]
    predicted = predictions[i, 0]
    prob = probabilities[i, 0]
    status = "✅" if predicted == expected else "❌"
    
    print(f"{status} Input: {X_xor[i]} → Predicted: {predicted}, Expected: {expected} (Confidence: {prob:.3f})")

# ========== EXERCISE 5 SOLUTIONS (Visualization & Analysis) ==========
section_header("Exercise 5 Solutions: Visualization & Analysis")

print("📝 SOLUTION - Advanced Visualization Functions")

def plot_decision_boundary_solution(network, X, y, title="Decision Boundary"):
    """Solution for plotting decision boundary"""
    plt.figure(figsize=(10, 8))
    
    # Create mesh
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Get predictions for mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    _, mesh_predictions = network.predict(mesh_points)
    mesh_predictions = mesh_predictions.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, mesh_predictions, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Network Output Probability')
    
    # Plot training points
    colors = ['red' if label[0] == 0 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2)
    
    # Add labels
    for i in range(len(X)):
        plt.annotate(f'{y[i, 0]}', (X[i, 0], X[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', color='white')
    
    plt.title(title)
    plt.xlabel('Input A')
    plt.ylabel('Input B')
    plt.grid(True, alpha=0.3)
    return plt.gcf()

# Demonstrate the solution
fig = plot_decision_boundary_solution(xor_network, X_xor, y_xor, "XOR Decision Boundary - Solution")
plt.show()

print("✅ Decision boundary visualization complete!")
print("   Explanation: Shows how the network divides input space")
print("   Red regions predict 0, blue regions predict 1")

def analyze_network_internals_solution(network, X, y):
    """Solution for analyzing network internals"""
    print("🔍 NETWORK INTERNAL ANALYSIS:")
    
    # Forward pass to get internal activations
    network.forward(X)
    
    print(f"\n📊 Final Weights and Biases:")
    print(f"W1 (input→hidden):\n{network.W1}")
    print(f"b1 (hidden biases): {network.b1}")
    print(f"W2 (hidden→output):\n{network.W2}")
    print(f"b2 (output bias): {network.b2}")
    
    print(f"\n🧠 Hidden Layer Activations:")
    print("Input → Hidden Neuron 1, Hidden Neuron 2")
    for i in range(len(X)):
        print(f"{X[i]} → {network.a1[i, 0]:.3f}, {network.a1[i, 1]:.3f}")
    
    # Analyze what each hidden neuron learned
    print(f"\n🎯 Hidden Neuron Interpretation:")
    
    # Check correlation with input patterns
    patterns = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
    
    print("Hidden Neuron 1 activations:", [f"{network.a1[i, 0]:.3f}" for i in range(4)])
    print("Hidden Neuron 2 activations:", [f"{network.a1[i, 1]:.3f}" for i in range(4)])
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.bar(patterns, network.a1[:, 0], alpha=0.7, color='orange')
    plt.title('Hidden Neuron 1 Activations')
    plt.ylabel('Activation Level')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.bar(patterns, network.a1[:, 1], alpha=0.7, color='green')
    plt.title('Hidden Neuron 2 Activations')
    plt.ylabel('Activation Level')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    final_output = network.a2.flatten()
    plt.bar(patterns, final_output, alpha=0.7, color='blue')
    plt.title('Final Output')
    plt.ylabel('Output Probability')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return network.a1, network.a2

# Analyze the trained network
hidden_activations, final_outputs = analyze_network_internals_solution(xor_network, X_xor, y_xor)

print("✅ Network analysis complete!")
print("   Each hidden neuron learned to detect specific input patterns")
print("   The output layer combines these patterns to produce XOR logic")

# ========== BONUS SOLUTIONS ==========
section_header("Bonus Solutions: Alternative Logic Gates")

def create_logic_gate_solution(gate_type="AND"):
    """Solution for implementing different logic gates"""
    
    # Define truth tables
    truth_tables = {
        "AND":  np.array([[0], [0], [0], [1]]),
        "OR":   np.array([[0], [1], [1], [1]]),
        "NAND": np.array([[1], [1], [1], [0]]),
        "NOR":  np.array([[1], [0], [0], [0]])
    }
    
    if gate_type not in truth_tables:
        raise ValueError(f"Unknown gate type: {gate_type}")
    
    # Training data (same inputs, different outputs)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = truth_tables[gate_type]
    
    # Create and train network
    print(f"🔧 Training {gate_type} gate...")
    network = XORSolution(learning_rate=3.0)
    network.train(X, y, epochs=300)
    
    # Test results
    predictions, probabilities = network.predict(X)
    
    print(f"\n✅ {gate_type} Gate Results:")
    for i in range(4):
        expected = y[i, 0]
        predicted = predictions[i, 0]
        prob = probabilities[i, 0]
        status = "✅" if predicted == expected else "❌"
        
        print(f"{status} Input: {X[i]} → Output: {predicted}, Expected: {expected} (Confidence: {prob:.3f})")
    
    return network

# Demonstrate different logic gates
print("🎯 BONUS: Implementing Other Logic Gates")

for gate in ["AND", "OR", "NAND"]:
    gate_network = create_logic_gate_solution(gate)
    print("-" * 50)

# ========== COMPREHENSIVE LEARNING SUMMARY ==========
section_header("Comprehensive Learning Summary")

print("""
🎓 LAB 1 COMPLETE SOLUTIONS SUMMARY

📚 What You Mastered:

1. NUMPY FUNDAMENTALS:
   ✅ Array creation and manipulation
   ✅ Matrix operations and broadcasting
   ✅ Vectorized computations

2. MATHEMATICAL FOUNDATIONS:
   ✅ Gradient descent optimization
   ✅ Numerical differentiation
   ✅ Activation functions (sigmoid)

3. SINGLE NEURON:
   ✅ Forward propagation
   ✅ Weight updates
   ✅ Basic learning mechanism

4. COMPLETE NEURAL NETWORK:
   ✅ Multi-layer architecture
   ✅ Backpropagation algorithm
   ✅ XOR problem solution
   ✅ Decision boundary visualization

5. ANALYSIS & VISUALIZATION:
   ✅ Network internals inspection
   ✅ Learning progress monitoring
   ✅ Decision boundary plotting

🧠 KEY INSIGHTS GAINED:

• Neural networks are universal function approximators
• Hidden layers enable non-linear pattern recognition
• Backpropagation teaches networks through error correction
• Matrix operations make neural networks computationally efficient
• Visualization helps understand what networks learn

🚀 NEXT STEPS:

• Lecture 3: Perceptron & Neural Basics
• Lecture 4: Activation Functions & Optimization
• Lab 2: Mathematical Foundations in Code
• Deeper networks and more complex problems

🏆 ACHIEVEMENT UNLOCKED: Neural Network Architect!

You've successfully built neural networks from scratch and solved
the famous XOR problem that launched the deep learning revolution.

Congratulations on completing this foundational lab! 🎉
""")

# Final visualization: Learning journey
plt.figure(figsize=(14, 6))

# Plot 1: Skill progression
skills = ['NumPy\nBasics', 'Math\nOps', 'Single\nNeuron', 'Full\nNetwork', 'Analysis &\nViz']
proficiency = [95, 90, 85, 95, 88]
colors = ['lightblue', 'lightgreen', 'orange', 'red', 'purple']

plt.subplot(1, 2, 1)
bars = plt.bar(skills, proficiency, color=colors, alpha=0.7, edgecolor='black')
plt.title('Lab 1: Skill Mastery Level', fontsize=14, fontweight='bold')
plt.ylabel('Proficiency (%)')
plt.ylim(0, 100)

# Add percentage labels on bars
for bar, prof in zip(bars, proficiency):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{prof}%', ha='center', va='bottom', fontweight='bold')

# Plot 2: Neural network complexity progression
plt.subplot(1, 2, 2)
exercises = ['Ex 1:\nArrays', 'Ex 2:\nMath', 'Ex 3:\nSingle\nNeuron', 'Ex 4:\nFull Net', 'Ex 5:\nAnalysis']
complexity = [2, 4, 6, 10, 8]
plt.plot(exercises, complexity, 'o-', linewidth=3, markersize=10, color='red', alpha=0.7)
plt.fill_between(range(len(exercises)), complexity, alpha=0.3, color='red')
plt.title('Complexity Progression', fontsize=14, fontweight='bold')
plt.ylabel('Complexity Level')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("🎯 LAB 1 SOLUTIONS COMPLETE - READY FOR THE NEXT CHALLENGE!")
print("="*70)