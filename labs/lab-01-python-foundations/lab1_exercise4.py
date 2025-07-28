"""
=============================================================================
LAB 1 - EXERCISE 4: COMPLETE NEURAL NETWORK (XOR PROBLEM)
=============================================================================
Learning Objectives:
• Build a complete neural network from scratch
• Solve the famous XOR problem that puzzled early AI researchers
• Implement forward propagation and backpropagation
• Train a network and watch it learn in real-time

INSTRUCTIONS:
1. Run previous exercises first (setup, exercise 1)
2. Read the XOR problem explanation carefully
3. Complete the TODO sections step by step
4. Watch your network learn to solve XOR!

Time: 45 minutes (Most Important Exercise!)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from lab1_setup import section_header, check_answer, sigmoid, sigmoid_derivative, mean_squared_error

section_header("Complete Neural Network - XOR Problem", 4)

print("""
🧩 THE XOR PROBLEM - The Challenge That Started Deep Learning

🎯 What is XOR?
XOR (Exclusive OR) is like a picky light switch:
• Turn ON when EXACTLY ONE switch is pressed
• Turn OFF when BOTH switches are pressed OR when NEITHER is pressed

Truth Table:
Input A | Input B | Output (XOR)
   0    |    0    |     0      ← Both OFF → Light OFF
   0    |    1    |     1      ← One ON  → Light ON  
   1    |    0    |     1      ← One ON  → Light ON
   1    |    1    |     0      ← Both ON → Light OFF

🎭 Why was this a BIG problem?
• In the 1960s, simple perceptrons (single neurons) couldn't solve XOR
• This caused the first "AI Winter" - people lost faith in neural networks
• The solution required HIDDEN LAYERS - thus "deep learning" was born!

Today, you'll build the network that ended the AI Winter! 🚀
""")

# ========== PART 1: PREPARE THE XOR DATA ==========
section_header("Part 1: Prepare XOR Training Data")

print("📊 Creating XOR dataset...")

# XOR training data
X_train = np.array([
    [0, 0],  # Input: both switches OFF
    [0, 1],  # Input: first OFF, second ON
    [1, 0],  # Input: first ON, second OFF  
    [1, 1]   # Input: both switches ON
])

y_train = np.array([
    [0],  # Output: light OFF
    [1],  # Output: light ON
    [1],  # Output: light ON
    [0]   # Output: light OFF
])

print(f"Training inputs (X):\n{X_train}")
print(f"Training outputs (y):\n{y_train}")

# Visualize the XOR problem
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = ['red', 'blue', 'blue', 'red']
for i in range(4):
    plt.scatter(X_train[i, 0], X_train[i, 1], c=colors[i], s=200, alpha=0.7)
    plt.annotate(f'Output: {y_train[i, 0]}', 
                (X_train[i, 0], X_train[i, 1]), 
                xytext=(10, 10), textcoords='offset points')

plt.xlabel('Input A')
plt.ylabel('Input B') 
plt.title('XOR Problem Visualization')
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

plt.subplot(1, 2, 2)
# Show why single line can't separate XOR
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
plt.contour(xx, yy, xx + yy, levels=[0.5, 1.5], colors=['gray'], linestyles=['--'])
for i in range(4):
    plt.scatter(X_train[i, 0], X_train[i, 1], c=colors[i], s=200, alpha=0.7)

plt.xlabel('Input A')
plt.ylabel('Input B')
plt.title('Why Single Line Cannot Solve XOR')
plt.text(0.5, 0.2, 'No single straight line can\nseparate red from blue points!', 
         ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

plt.tight_layout()
plt.show()

print("""
🔍 Key Insight: The XOR problem is NOT linearly separable!
• You cannot draw a single straight line to separate the red dots from blue dots
• This is why we need HIDDEN LAYERS - they create multiple decision boundaries
• Our neural network will learn to combine multiple lines to solve this puzzle!
""")

# ========== PART 2: DESIGN THE NEURAL NETWORK ==========
section_header("Part 2: Design Our XOR-Solving Neural Network")

print("""
🏗️ NETWORK ARCHITECTURE - Our XOR Solver Blueprint

Our network structure:
┌─────────┐    ┌─────────────────┐    ┌─────────┐
│ Input   │    │ Hidden Layer    │    │ Output  │
│ Layer   │───▶│ (2 neurons)     │───▶│ Layer   │
│(2 nodes)│    │ Sigmoid         │    │(1 node) │
└─────────┘    └─────────────────┘    └─────────┘
     ▲                   ▲                   ▲
  Features           Decision            Final
(Switch A,B)        Boundaries          Answer

Think of it like a two-step decision process:
1. Hidden layer: "Are we in a special XOR pattern?"
2. Output layer: "Based on the pattern, turn light ON or OFF?"
""")

class XORNeuralNetwork:
    def __init__(self, learning_rate=1.0):
        """
        Initialize our XOR-solving neural network
        
        Architecture:
        - Input layer: 2 neurons (for A and B inputs)
        - Hidden layer: 2 neurons (to create decision boundaries)
        - Output layer: 1 neuron (for final XOR result)
        """
        self.learning_rate = learning_rate
        
        # Initialize weights randomly (small values)
        # Think of weights as "connection strengths" between neurons
        np.random.seed(42)  # For reproducible results
        
        # Weights from input to hidden layer (2×2 matrix)
        # Each input connects to each hidden neuron
        self.W1 = np.random.uniform(-1, 1, (2, 2))
        
        # Biases for hidden layer (2 values)  
        # Think of bias as "how easily the neuron gets excited"
        self.b1 = np.random.uniform(-1, 1, (2,))
        
        # Weights from hidden to output layer (2×1 matrix)
        # Each hidden neuron connects to the output
        self.W2 = np.random.uniform(-1, 1, (2, 1))
        
        # Bias for output layer (1 value)
        self.b2 = np.random.uniform(-1, 1, (1,))
        
        # Store training history for visualization
        self.loss_history = []
        self.accuracy_history = []
        
        print(f"🧠 Network initialized!")
        print(f"W1 (input→hidden): {self.W1.shape}")
        print(f"b1 (hidden bias): {self.b1.shape}") 
        print(f"W2 (hidden→output): {self.W2.shape}")
        print(f"b2 (output bias): {self.b2.shape}")
    
    def forward(self, X):
        """
        Forward propagation - How information flows through the network
        
        Like water flowing through pipes:
        Input → Hidden Layer → Output Layer
        """
        # Step 1: Input to Hidden Layer
        # Matrix multiplication + bias addition
        self.z1 = X @ self.W1 + self.b1  # Linear combination
        self.a1 = sigmoid(self.z1)        # Activation (decision making)
        
        # Step 2: Hidden to Output Layer  
        self.z2 = self.a1 @ self.W2 + self.b2  # Linear combination
        self.a2 = sigmoid(self.z2)              # Final activation
        
        return self.a2
    
    def backward(self, X, y, output):
        """
        Backpropagation - How the network learns from mistakes
        
        Like a student reviewing wrong answers:
        1. Calculate how wrong we were (error)
        2. Figure out which weights caused the error
        3. Adjust weights to reduce future errors
        """
        m = X.shape[0]  # Number of training examples
        
        # Step 1: Calculate output layer error
        # "How wrong was our final answer?"
        dZ2 = output - y  # Error in predictions
        dW2 = (self.a1.T @ dZ2) / m  # How much each hidden→output weight contributed
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # How much the output bias contributed
        
        # Step 2: Calculate hidden layer error (backpropagate)
        # "Which hidden neurons caused the output error?"
        dA1 = dZ2 @ self.W2.T  # Error flowing back to hidden layer
        dZ1 = dA1 * sigmoid_derivative(self.z1)  # Adjust for sigmoid derivative
        dW1 = (X.T @ dZ1) / m  # How much each input→hidden weight contributed
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # How much hidden biases contributed
        
        # Step 3: Update weights (gradient descent)
        # "Adjust weights to reduce errors"
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2.T
        self.W1 -= self.learning_rate * dW1  
        self.b1 -= self.learning_rate * db1.flatten()
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the network to solve XOR
        
        Like practicing piano:
        - Play the piece (forward pass)
        - Notice mistakes (calculate loss)
        - Adjust technique (backpropagation)
        - Repeat until perfect!
        """
        print(f"🎯 Training network for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Forward pass - make predictions
            output = self.forward(X)
            
            # Calculate loss (how wrong we are)
            loss = mean_squared_error(y, output)
            self.loss_history.append(loss)
            
            # Calculate accuracy (% of correct predictions)
            predictions = (output > 0.5).astype(int)
            accuracy = np.mean(predictions == y) * 100
            self.accuracy_history.append(accuracy)
            
            # Backward pass - learn from mistakes
            self.backward(X, y, output)
            
            # Print progress every 100 epochs
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:4d}: Loss = {loss:.6f}, Accuracy = {accuracy:.1f}%")
        
        print(f"🎉 Training complete!")
        final_output = self.forward(X)
        final_predictions = (final_output > 0.5).astype(int)
        final_accuracy = np.mean(final_predictions == y) * 100
        print(f"Final accuracy: {final_accuracy:.1f}%")
        
        return self.loss_history, self.accuracy_history
    
    def predict(self, X):
        """Make predictions on new data"""
        output = self.forward(X)
        return (output > 0.5).astype(int), output

# ========== PART 3: TRAIN THE NETWORK ==========
section_header("Part 3: Train the Network to Solve XOR")

print("🚀 Creating and training our XOR neural network...")

# Create the network
network = XORNeuralNetwork(learning_rate=5.0)

print("\n📊 Initial predictions (before training):")
initial_output = network.forward(X_train)
initial_predictions = (initial_output > 0.5).astype(int)

for i in range(4):
    print(f"Input: {X_train[i]} → Predicted: {initial_predictions[i, 0]}, Expected: {y_train[i, 0]} (Raw: {initial_output[i, 0]:.3f})")

print("\n" + "="*50)
print("🎯 YOUR TURN - TODO 1:")
print("📝 TODO 1: Train the network and watch it learn!")

# TODO: Train the network
print("Uncomment and run the training:")
print("# loss_history, accuracy_history = network.train(X_train, y_train, epochs=1000)")

# STUDENT CODE HERE - Uncomment the line below
# loss_history, accuracy_history = network.train(X_train, y_train, epochs=1000)

# For demonstration, let's train it
loss_history, accuracy_history = network.train(X_train, y_train, epochs=1000)

print("\n📊 Final predictions (after training):")
final_predictions, final_probabilities = network.predict(X_train)

for i in range(4):
    status = "✅" if final_predictions[i, 0] == y_train[i, 0] else "❌"
    print(f"{status} Input: {X_train[i]} → Predicted: {final_predictions[i, 0]}, Expected: {y_train[i, 0]} (Confidence: {final_probabilities[i, 0]:.3f})")

# ========== PART 4: VISUALIZE THE LEARNING PROCESS ==========
section_header("Part 4: Visualize How the Network Learned")

# Plot training progress
plt.figure(figsize=(15, 10))

# Plot 1: Loss over time
plt.subplot(2, 3, 1)
plt.plot(loss_history, 'b-', linewidth=2)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 2: Accuracy over time
plt.subplot(2, 3, 2)
plt.plot(accuracy_history, 'g-', linewidth=2)
plt.title('Training Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.ylim(0, 105)

# Plot 3: Decision boundary visualization
plt.subplot(2, 3, 3)
# Create a grid of points to test
h = 0.02
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Get network predictions for all grid points
mesh_points = np.c_[xx.ravel(), yy.ravel()]
_, mesh_predictions = network.predict(mesh_points)
mesh_predictions = mesh_predictions.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, mesh_predictions, levels=50, alpha=0.6, cmap='RdYlBu')
plt.colorbar(label='Network Output')

# Plot training points
colors = ['red', 'blue', 'blue', 'red']
labels = ['OFF (0)', 'ON (1)', 'ON (1)', 'OFF (0)']
for i in range(4):
    plt.scatter(X_train[i, 0], X_train[i, 1], c=colors[i], s=200, 
               edgecolors='black', linewidth=2, alpha=0.9)
    plt.annotate(labels[i], (X_train[i, 0], X_train[i, 1]), 
                xytext=(10, 10), textcoords='offset points', fontweight='bold')

plt.title('Learned Decision Boundary')
plt.xlabel('Input A')
plt.ylabel('Input B')

# Plot 4: Network weights visualization
plt.subplot(2, 3, 4)
weights_combined = np.concatenate([network.W1.flatten(), network.W2.flatten(), 
                                  network.b1.flatten(), network.b2.flatten()])
plt.bar(range(len(weights_combined)), weights_combined, alpha=0.7)
plt.title('Final Network Weights & Biases')
plt.xlabel('Parameter Index')
plt.ylabel('Weight Value')
plt.grid(True, alpha=0.3)

# Plot 5: Hidden layer activations
plt.subplot(2, 3, 5)
hidden_activations = network.a1
plt.imshow(hidden_activations.T, cmap='viridis', aspect='auto')
plt.colorbar(label='Activation Level')
plt.title('Hidden Layer Activations')
plt.xlabel('Training Sample')
plt.ylabel('Hidden Neuron')
plt.xticks(range(4), ['[0,0]', '[0,1]', '[1,0]', '[1,1]'])

# Plot 6: Learning curve analysis
plt.subplot(2, 3, 6)
plt.plot(loss_history, 'b-', label='Loss', alpha=0.7)
plt.plot(np.array(accuracy_history)/100, 'g-', label='Accuracy/100', alpha=0.7)
plt.title('Learning Progress')
plt.xlabel('Epoch')  
plt.ylabel('Normalized Value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========== PART 5: UNDERSTANDING WHAT THE NETWORK LEARNED ==========
section_header("Part 5: Understanding What the Network Learned")

print("""
🧠 NETWORK ANALYSIS - What Did Our AI Brain Learn?

Let's peek inside the trained network to understand how it solved XOR:
""")

print("🔍 Final Network Parameters:")
print(f"Input→Hidden Weights (W1):\n{network.W1}")
print(f"Hidden→Output Weights (W2):\n{network.W2}")
print(f"Hidden Biases (b1): {network.b1}")
print(f"Output Bias (b2): {network.b2}")

print("\n🎭 Hidden Layer Analysis:")
print("Let's see what each hidden neuron learned to detect:")

for i in range(4):
    input_val = X_train[i]
    z1 = input_val @ network.W1 + network.b1
    a1 = sigmoid(z1)
    
    print(f"\nInput {input_val}:")
    print(f"  Hidden Neuron 1: {a1[0]:.3f} ({'Active' if a1[0] > 0.5 else 'Inactive'})")
    print(f"  Hidden Neuron 2: {a1[1]:.3f} ({'Active' if a1[1] > 0.5 else 'Inactive'})")

print("""
🎯 Key Insights:
• The hidden neurons learned to detect different patterns in the input
• One neuron might detect "at least one input is ON"
• Another might detect "both inputs are ON" 
• The output layer combines these patterns to produce XOR logic

This is the power of hidden layers - they learn to detect complex patterns
that single neurons cannot! 🚀
""")

# ========== PART 6: TEST ON NEW DATA ==========
section_header("Part 6: Test the Trained Network")

print("🧪 Testing our network on the original XOR problem:")

test_cases = [
    ([0, 0], "Both switches OFF"),
    ([0, 1], "First OFF, Second ON"),
    ([1, 0], "First ON, Second OFF"), 
    ([1, 1], "Both switches ON")
]

print("\n🎯 YOUR TURN - TODO 2:")
print("📝 TODO 2: Test the network with new inputs")

for inputs, description in test_cases:
    test_input = np.array([inputs])
    prediction, probability = network.predict(test_input)
    
    expected = inputs[0] ^ inputs[1]  # XOR logic in Python
    correct = "✅" if prediction[0, 0] == expected else "❌"
    
    print(f"{correct} {description}")
    print(f"   Input: {inputs} → Output: {prediction[0, 0]} (Confidence: {probability[0, 0]:.3f})")
    print(f"   Expected XOR result: {expected}")
    print()

# Interactive testing section
print("🎮 Interactive Testing:")
print("Try testing with your own values!")

# TODO: Students can modify these values
print("\n📝 TODO 3: Test with custom inputs (modify the values below)")
custom_tests = [
    [0.1, 0.9],  # Almost [0, 1]
    [0.9, 0.1],  # Almost [1, 0]  
    [0.9, 0.9],  # Almost [1, 1]
    [0.1, 0.1]   # Almost [0, 0]
]

for test_input in custom_tests:
    test_array = np.array([test_input])
    prediction, probability = network.predict(test_array)
    
    print(f"Input: {test_input} → Output: {prediction[0, 0]} (Confidence: {probability[0, 0]:.3f})")

# ========== EXERCISE 4 SUMMARY ==========
section_header("Exercise 4 Summary - You Built a Neural Network!")

print("""
🎉 INCREDIBLE ACHIEVEMENT! You just solved the XOR problem!

🏆 What you accomplished:
✅ Built a complete neural network from scratch
✅ Implemented forward propagation (information flow)
✅ Implemented backpropagation (learning mechanism)
✅ Solved the famous XOR problem that stumped early AI
✅ Visualized how neural networks learn and make decisions

🧠 Key Concepts Mastered:
• Hidden layers create complex decision boundaries
• Backpropagation teaches networks from their mistakes
• Sigmoid activation enables non-linear learning
• Matrix operations make neural networks efficient

🌟 Historical Significance:
You just recreated the breakthrough that:
• Ended the first AI Winter in the 1980s
• Proved neural networks could solve non-linear problems
• Laid the foundation for modern deep learning
• Led to today's AI revolution!

🚀 What's Next:
• Exercise 5: Visualization & Analysis
• Lecture 3: Perceptron & Neural Basics
• More complex problems and deeper networks

You're now officially a neural network architect! 🏗️🧠
""")

# Final challenge
print("\n🎯 BONUS CHALLENGE (Optional):")
print("Can you modify the network to solve other logic gates?")
print("Try changing the training data to solve:")
print("• AND gate: [0,0]→0, [0,1]→0, [1,0]→0, [1,1]→1")
print("• OR gate:  [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→1")
print("• NAND gate: [0,0]→1, [0,1]→1, [1,0]→1, [1,1]→0")

print("\n" + "="*70)
print("🎓 EXERCISE 4 COMPLETE - NEURAL NETWORK MASTER!")
print("="*70)