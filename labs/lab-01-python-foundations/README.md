\# Lab 1: Python Foundations \& First Neural Network



\*\*Course:\*\* Deep Learning Mastery | \*\*Instructor:\*\* Dr. Daya Shankar | \*\*Duration:\*\* 2 Hours



\[!\[Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dayashankar-ai/interactive-deep-learning-lectures/blob/main/labs/lab-01-python-foundations/lab1\_complete\_notebook.ipynb)



\## 🎯 Learning Objectives



By the end of this lab, you will:

\- ✅ Master NumPy operations essential for neural networks

\- ✅ Implement mathematical concepts from Lecture 2 (derivatives, gradient descent)

\- ✅ Build a single neuron from scratch

\- ✅ Create a complete neural network to solve the XOR problem

\- ✅ Visualize and analyze neural network learning



\## ⏰ Lab Structure (2 Hours)



| Exercise | Topic | Duration | Description |

|----------|-------|----------|-------------|

| \*\*Setup\*\* | Environment | 5 min | Install packages, verify setup |

| \*\*Exercise 1\*\* | NumPy Fundamentals | 25 min | Arrays, matrices, broadcasting |

| \*\*Exercise 2\*\* | Mathematical Operations | 20 min | Derivatives, gradient descent |

| \*\*Exercise 3\*\* | Single Neuron | 25 min | Build first artificial neuron |

| \*\*Exercise 4\*\* | Complete Neural Network | 45 min | XOR problem solution ⭐ |

| \*\*Exercise 5\*\* | Visualization \& Analysis | 20 min | Learning curves, decision boundaries |



\## 🚀 Quick Start Options



\### Option 1: Google Colab (Recommended)

```bash

\# Click the Colab badge above or this link:

\# https://colab.research.google.com/github/dayashankar-ai/interactive-deep-learning-lectures/blob/main/labs/lab-01-python-foundations/lab1\_complete\_notebook.ipynb



\# Then:

\# 1. File → Save a copy in Drive

\# 2. Runtime → Change runtime type → GPU

\# 3. Run all cells sequentially

```



\### Option 2: Individual Python Files

```bash

\# Copy each Python file to Google Colab or your local environment

\# Run in this order:

python lab1\_setup.py          # First - environment setup

python lab1\_exercise1.py      # NumPy fundamentals

python lab1\_exercise2.py      # Mathematical operations  

python lab1\_exercise3.py      # Single neuron

python lab1\_exercise4.py      # Complete neural network

python lab1\_exercise5.py      # Visualization

python lab1\_solutions.py      # Check your work

```



\### Option 3: Local Development

```bash

\# Clone repository

git clone https://github.com/dayashankar-ai/interactive-deep-learning-lectures.git

cd interactive-deep-learning-lectures/labs/lab-01-python-foundations



\# Install requirements

pip install numpy matplotlib pandas



\# Start with setup

python lab1\_setup.py

```



\## 📁 File Structure



```

lab-01-python-foundations/

├── README.md                     # This file

├── lab1\_setup.py                 # Environment setup \& imports

├── lab1\_exercise1.py             # NumPy fundamentals

├── lab1\_exercise2.py             # Mathematical operations

├── lab1\_exercise3.py             # Single neuron implementation

├── lab1\_exercise4.py             # Complete neural network (XOR)

├── lab1\_exercise5.py             # Visualization \& analysis

├── lab1\_solutions.py             # Complete solutions

├── lab1\_complete\_notebook.ipynb  # Jupyter notebook version

└── assets/

&nbsp;   ├── images/                   # Diagrams and screenshots

&nbsp;   └── screenshots/              # Expected outputs

```



\## 🧠 Key Concepts Covered



\### Exercise 1: NumPy Fundamentals

\- \*\*Array Operations\*\*: Creation, indexing, slicing

\- \*\*Matrix Multiplication\*\*: The heart of neural networks

\- \*\*Broadcasting\*\*: NumPy's superpower for different shapes

\- \*\*Vectorization\*\*: Efficient batch processing



\*\*Real-World Connection\*\*: \*Every neural network operation uses these NumPy concepts\*



\### Exercise 2: Mathematical Operations

\- \*\*Numerical Derivatives\*\*: Computing gradients with code

\- \*\*Gradient Descent\*\*: How neural networks learn

\- \*\*Activation Functions\*\*: Sigmoid and its derivative

\- \*\*Optimization\*\*: Finding minimum error



\*\*Real-World Connection\*\*: \*This is the math that makes AI learn from mistakes\*



\### Exercise 3: Single Neuron

\- \*\*Neuron Architecture\*\*: Weights, bias, activation

\- \*\*Forward Pass\*\*: How neurons process information

\- \*\*Learning Rule\*\*: Weight updates from errors

\- \*\*Logic Gates\*\*: OR gate implementation



\*\*Real-World Connection\*\*: \*Building blocks of all AI systems\*



\### Exercise 4: Complete Neural Network ⭐

\- \*\*XOR Problem\*\*: Historical significance in AI

\- \*\*Multi-layer Architecture\*\*: Hidden layers enable complex learning

\- \*\*Backpropagation\*\*: The breakthrough algorithm

\- \*\*Training Loop\*\*: Iterative improvement process



\*\*Real-World Connection\*\*: \*The network that ended the AI Winter and launched deep learning\*



\### Exercise 5: Visualization \& Analysis

\- \*\*Learning Curves\*\*: Monitoring training progress

\- \*\*Decision Boundaries\*\*: What the network learned

\- \*\*Weight Evolution\*\*: How parameters change during training

\- \*\*Performance Analysis\*\*: Accuracy and error metrics



\*\*Real-World Connection\*\*: \*Understanding and debugging AI models\*



\## 🎯 Expected Outcomes



After completing this lab, you should be able to:



\- \[ ] \*\*Explain\*\* why matrix operations are fundamental to neural networks

\- \[ ] \*\*Implement\*\* gradient descent from scratch

\- \[ ] \*\*Build\*\* a functioning neural network without frameworks

\- \[ ] \*\*Solve\*\* the XOR problem that stumped early AI researchers

\- \[ ] \*\*Visualize\*\* how neural networks learn and make decisions

\- \[ ] \*\*Debug\*\* neural network training issues



\## 🏆 Challenge Problems (Optional)



1\. \*\*Architecture Experiment\*\*: Modify the XOR network to use different numbers of hidden neurons (2, 4, 8, 16). How does this affect learning speed and final accuracy?



2\. \*\*Logic Gate Generator\*\*: Create networks to solve AND, OR, NAND, and NOR gates. Which ones are harder to learn and why?



3\. \*\*Custom Dataset\*\*: Generate a circular classification dataset (points inside/outside a circle) and train your network to solve it.



4\. \*\*Learning Rate Analysis\*\*: Test different learning rates (0.01, 0.1, 1.0, 10.0) and observe their effects on convergence.



\## 🛠️ Troubleshooting Guide



\### Common Issues \& Solutions



\*\*Issue\*\*: "Runtime disconnected in Colab"

\- \*\*Solution\*\*: Runtime → Reconnect. Save work frequently!



\*\*Issue\*\*: "Matrix dimension mismatch"  

\- \*\*Solution\*\*: Check array shapes with `.shape`. Use `np.reshape()` if needed.



\*\*Issue\*\*: "Loss not decreasing"

\- \*\*Solution\*\*: Try different learning rates (0.01, 0.1, 1.0). Check gradient calculations.



\*\*Issue\*\*: "Overflow/underflow errors"

\- \*\*Solution\*\*: Use `np.clip()` to prevent extreme values in exponential functions.



\### Getting Help



\- \*\*Solutions\*\*: Check `lab1\_solutions.py` for complete implementations

\- \*\*Review\*\*: Revisit Lecture 2 (Mathematical Foundations)

\- \*\*Discussion\*\*: Course forum or office hours

\- \*\*Email\*\*: dayashankar.ai@gmail.com



\## 📊 Assessment Criteria



Your lab will be evaluated on:



| Component | Weight | Criteria |

|-----------|--------|----------|

| \*\*Code Implementation\*\* | 40% | Correct solutions, clean code, proper comments |

| \*\*Neural Network\*\* | 30% | XOR problem solved, good performance |

| \*\*Visualizations\*\* | 15% | Clear plots with proper labels |

| \*\*Analysis\*\* | 15% | Thoughtful observations and insights |



\## 🔗 Related Content



\- \*\*Prerequisites\*\*: \[Lecture 1: History](../../lectures/lecture01-history-foundations.html) | \[Lecture 2: Math Foundations](../../lectures/lecture02-mathematical-foundations.html)

\- \*\*Next\*\*: \[Lecture 3: Perceptron Basics](../../lectures/lecture03-perceptron-basics.html)

\- \*\*Future Labs\*\*: Lab 2 (Mathematical Foundations in Code) | Lab 3 (Building Neural Networks)



\## 🎉 Achievement Unlocked



Upon completion, you'll have:

\- \*\*Built\*\* your first neural network from scratch

\- \*\*Solved\*\* the XOR problem that launched deep learning

\- \*\*Mastered\*\* the mathematical foundations of AI

\- \*\*Gained\*\* confidence to tackle more complex architectures



\*\*You're now officially a Neural Network Architect!\*\* 🏗️🧠



---



\*\*Created by Dr. Daya Shankar\*\* | Dean of Sciences, Woxsen University | Founder, VaidyaAI



🌐 \[Personal Website](https://www.dayashankar.com) | 🏥 \[VaidyaAI](https://vaidyaai.com) | 🎓 \[Woxsen University](https://woxsen.edu.in)

