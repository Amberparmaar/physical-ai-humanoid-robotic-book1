---
sidebar_label: Physics Simulation in Gazebo
---

# Physics Simulation in Gazebo

## Physics Engines

Gazebo supports multiple physics engines, each with its own strengths:

### Open Dynamics Engine (ODE)
- **Strengths**: Fast simulation, good for basic rigid body dynamics
- **Use Cases**: Simple robots, basic manipulation tasks
- **Limitations**: Less accurate for complex contacts

### Bullet Physics
- **Strengths**: More accurate contact simulation, good for complex interactions
- **Use Cases**: Multi-contact scenarios, more realistic simulations
- **Limitations**: Slightly slower than ODE

### Simbody
- **Strengths**: Highly accurate for articulated systems
- **Use Cases**: Complex humanoid robots with many joints
- **Limitations**: More computationally intensive

## Physics Parameters

### World Properties
- **Gravity**: Default value is 9.8 m/s² downward acceleration
- **Real Time Update Rate**: Controls simulation update frequency
- **Max Step Size**: Maximum physics step size in seconds
- **Real Time Factor**: Desired speedup relative to real time

### Material Properties
- **Mu (Friction Coefficient)**: Determines tangential friction between surfaces
- **Mu2**: Secondary friction coefficient (for anisotropic friction)
- **Slip1/Slip2**: Inverse of stiffness for tangential forces
- **Restitution Coefficient**: Determines bounciness of contacts
- **Soft ERP/CFM**: Error reduction and constraint force mixing parameters

## Collision Detection

Gazebo employs multiple collision detection strategies:

### Broad Phase
- Quick elimination of non-colliding pairs
- Uses bounding volume hierarchies

### Narrow Phase
- Precise collision detection
- Calculates contact points, normals, and depths

### Contact Processing
- Computes contact forces based on physics parameters
- Handles multiple simultaneous contacts

## Physics Configuration for Humanoid Robots

### Joint Simulation
- **Joint Limits**: Define position, velocity, and effort constraints
- **Spring Damping**: Simulate compliance in joints (important for humanoid safety)
- **Friction Models**: Static, Coulomb, and viscous friction parameters

### Balance and Stability
- **Center of Mass**: Accurate placement crucial for stable walking
- **Inertia Tensors**: Proper moment of inertia values for realistic dynamics
- **Contact Stabilization**: Parameters to prevent jittering during contact

### Ground Interaction
- **Foot Contact**: Accurate modeling of foot-ground interaction for walking
- **Slip Parameters**: Realistic foot slipping behavior
- **Ground Compliance**: Soft ground effects for more realistic interaction

## Performance Considerations

### Step Size Optimization
- Smaller step sizes provide more accurate simulation but require more computation
- Balance accuracy requirements with computational constraints

### Contact Parameters
- Adjust ERP (Error Reduction Parameter) and CFM (Constraint Force Mixing) for stable contacts
- Too low ERP may cause instability; too high may cause artificial stiffness

### Model Complexity
- Simplify collision geometries where high accuracy isn't needed
- Use appropriate geometric primitives (boxes, cylinders, spheres) when possible

## Tuning Physics Parameters

### Stability Tuning
1. Start with default parameters and gradually adjust
2. Increase ERP values if joints are unstable
3. Decrease CFM values for stiffer contacts
4. Adjust damping coefficients to match real-world behavior

### Accuracy vs. Performance
- For real-time applications, optimize for performance while maintaining required accuracy
- For offline simulation and data generation, prioritize accuracy
- Use fixed-step simulation for reproducible results

## Advanced Physics Features

### Fluid Simulation
- Simulate interaction with water or other fluids
- Useful for humanoid robots that need to navigate wet environments

### Granular Materials
- Simulate sand, gravel, or other granular materials
- Important for humanoid robots operating in outdoor environments

### Flexible Bodies
- Simulate deformation of non-rigid objects
- Useful for soft robotics research

Understanding physics simulation in Gazebo is crucial for creating realistic and stable humanoid robotics simulations that closely match real-world behavior.