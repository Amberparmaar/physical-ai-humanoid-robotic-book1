---
sidebar_position: 5
title: Unity Robotics Simulation
---

# Unity Robotics Simulation

In this chapter, we'll explore Unity as a powerful platform for robotics simulation, particularly for creating photorealistic environments and advanced perception systems. Unity provides capabilities that complement traditional physics-based simulators like Gazebo, especially for computer vision and machine learning applications.

## Understanding Unity for Robotics

Unity is a versatile game engine that has been adapted for robotics simulation through the Unity Robotics Hub. Key advantages include:

- **Photorealistic rendering**: High-quality graphics for computer vision training
- **Flexible environments**: Ability to create complex, detailed worlds
- **Physics simulation**: Built-in physics engine with realistic interactions
- **Cross-platform deployment**: Deploy simulations across multiple platforms
- **Asset ecosystem**: Extensive library of 3D models and environments
- **Scripting flexibility**: C# scripting for custom robot behaviors and simulation logic

## Unity Robotics Hub Components

The Unity Robotics Hub provides several key components:

1. **Unity Robotics Package**: Core tools for ROS integration
2. **Unity Perception Package**: Tools for synthetic data generation
3. **Unity Simulation Package**: Tools for distributed simulation
4. **Robotics Examples**: Sample projects demonstrating best practices

### Basic Unity Robotics Setup

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    [SerializeField]
    private float linearVelocity = 1.0f;
    [SerializeField]
    private float angularVelocity = 1.0f;

    private ROSConnection ros;
    private string cmdVelTopic = "/cmd_vel";

    // Robot components
    private Rigidbody rb;
    private float currentLinear = 0f;
    private float currentAngular = 0f;

    void Start()
    {
        // Connect to ROS
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TwistMsg>(cmdVelTopic);

        rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        // Apply movement based on current commands
        if (rb != null)
        {
            // Convert differential drive commands to movement
            Vector3 forward = transform.forward * currentLinear * Time.deltaTime;
            transform.Translate(forward, Space.World);

            Vector3 rotation = new Vector3(0, currentAngular, 0) * Time.deltaTime;
            transform.Rotate(rotation, Space.Self);
        }
    }

    void OnMessageReceived(TwistMsg cmd)
    {
        // Process incoming velocity commands
        currentLinear = (float)cmd.linear.x;
        currentAngular = (float)cmd.angular.z;
    }

    void OnEnable()
    {
        ros.Subscribe<TwistMsg>(cmdVelTopic, OnMessageReceived);
    }

    void OnDisable()
    {
        ros.Unsubscribe<TwistMsg>(cmdVelTopic, OnMessageReceived);
    }
}
```

## Creating Robot Models in Unity

### Unity Robot Structure

```csharp
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class URDFJoint : MonoBehaviour
{
    [Header("Joint Configuration")]
    public JointType jointType = JointType.Revolute;
    public Vector3 axis = Vector3.up;
    public float lowerLimit = -45f;
    public float upperLimit = 45f;
    public float maxVelocity = 10f;
    public float maxEffort = 100f;

    [Header("Dynamics")]
    public float damping = 0.1f;
    public float stiffness = 10f;

    private Joint joint;
    private ConfigurableJoint configJoint;

    void Start()
    {
        SetupJoint();
    }

    void SetupJoint()
    {
        switch (jointType)
        {
            case JointType.Revolute:
            case JointType.Continuous:
                configJoint = gameObject.AddComponent<ConfigurableJoint>();
                ConfigureRevoluteJoint();
                break;
            case JointType.Prismatic:
                configJoint = gameObject.AddComponent<ConfigurableJoint>();
                ConfigurePrismaticJoint();
                break;
            case JointType.Fixed:
                // Fixed joint - no movement
                break;
        }
    }

    void ConfigureRevoluteJoint()
    {
        if (configJoint != null)
        {
            configJoint.axis = axis;
            configJoint.secondaryAxis = Vector3.Cross(axis, Vector3.up);

            // Set angular limits
            if (jointType == JointType.Revolute)
            {
                JointLimitLimitSpring limit = configJoint.angularXLimit;
                limit.limit = upperLimit;
                configJoint.angularXLimit = limit;

                configJoint.angularXMotion = ConfigurableJointMotion.Limited;
            }
            else
            {
                // Continuous joint - no limits
                configJoint.angularXMotion = ConfigurableJointMotion.Free;
            }

            // Configure drive for actuation
            JointDrive drive = configJoint.slerpDrive;
            drive.positionSpring = stiffness;
            drive.positionDamper = damping;
            drive.maximumForce = maxEffort;
            configJoint.slerpDrive = drive;
        }
    }

    void ConfigurePrismaticJoint()
    {
        if (configJoint != null)
        {
            configJoint.axis = axis;

            // Set linear limits
            if (jointType == JointType.Prismatic)
            {
                SoftJointLimit limit = configJoint.linearLimit;
                limit.limit = upperLimit;
                configJoint.linearLimit = limit;

                configJoint.linearLimitMotion = ConfigurableJointMotion.Limited;
            }
            else
            {
                configJoint.linearLimitMotion = ConfigurableJointMotion.Free;
            }

            // Configure drive for actuation
            JointDrive drive = configJoint.xDrive;
            drive.positionSpring = stiffness;
            drive.positionDamper = damping;
            drive.maximumForce = maxEffort;
            configJoint.xDrive = drive;
        }
    }

    public enum JointType
    {
        Fixed,
        Revolute,
        Continuous,
        Prismatic
    }
}
```

## Sensor Simulation in Unity

### Camera Sensor Implementation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections;

public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Configuration")]
    [SerializeField] private Camera sensorCamera;
    [SerializeField] private int imageWidth = 640;
    [SerializeField] private int imageHeight = 480;
    [SerializeField] private float updateRate = 30f;
    [SerializeField] private string imageTopic = "/camera/image_raw";

    private ROSConnection ros;
    private RenderTexture renderTexture;
    private Texture2D tempTexture;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Create render texture for camera
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        sensorCamera.targetTexture = renderTexture;

        // Create temporary texture for reading pixels
        tempTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        updateInterval = 1f / updateRate;
        lastUpdateTime = 0f;
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishCameraImage();
            lastUpdateTime = Time.time;
        }
    }

    void PublishCameraImage()
    {
        // Set active render texture
        RenderTexture.active = renderTexture;

        // Read pixels from render texture
        tempTexture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        tempTexture.Apply();

        // Convert to ROS image message
        ImageMsg imageMsg = CreateImageMessage(tempTexture);

        // Publish to ROS
        ros.Publish(imageTopic, imageMsg);
    }

    ImageMsg CreateImageMessage(Texture2D texture)
    {
        ImageMsg msg = new ImageMsg();

        // Set image properties
        msg.header = new std_msgs.HeaderMsg();
        msg.header.stamp = new TimeMsg(0, (uint)(Time.time * 1e9));
        msg.header.frame_id = transform.name;

        msg.height = (uint)texture.height;
        msg.width = (uint)texture.width;
        msg.encoding = "rgb8";
        msg.is_bigendian = false;
        msg.step = (uint)(texture.width * 3); // 3 bytes per pixel for RGB

        // Convert texture to byte array
        byte[] imageData = texture.EncodeToPNG();
        msg.data = imageData;

        return msg;
    }

    void OnDestroy()
    {
        if (renderTexture != null)
            RenderTexture.DestroyImmediate(renderTexture);
        if (tempTexture != null)
            Texture2D.DestroyImmediate(tempTexture);
    }
}
```

### LIDAR Sensor Simulation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class UnityLIDARSensor : MonoBehaviour
{
    [Header("LIDAR Configuration")]
    [SerializeField] private float minAngle = -90f;  // degrees
    [SerializeField] private float maxAngle = 90f;   // degrees
    [SerializeField] private int numRays = 720;
    [SerializeField] private float minRange = 0.1f;
    [SerializeField] private float maxRange = 30f;
    [SerializeField] private float updateRate = 10f;
    [SerializeField] private string scanTopic = "/scan";

    private ROSConnection ros;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        updateInterval = 1f / updateRate;
        lastUpdateTime = 0f;
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishLIDARScan();
            lastUpdateTime = Time.time;
        }
    }

    void PublishLIDARScan()
    {
        LaserScanMsg scanMsg = new LaserScanMsg();

        // Set header
        scanMsg.header = new std_msgs.HeaderMsg();
        scanMsg.header.stamp = new TimeMsg(0, (uint)(Time.time * 1e9));
        scanMsg.header.frame_id = transform.name;

        // Set scan parameters
        scanMsg.angle_min = minAngle * Mathf.Deg2Rad;
        scanMsg.angle_max = maxAngle * Mathf.Deg2Rad;
        scanMsg.angle_increment = (maxAngle - minAngle) * Mathf.Deg2Rad / numRays;
        scanMsg.time_increment = 0f; // Not applicable for Unity simulation
        scanMsg.scan_time = 1f / updateRate;
        scanMsg.range_min = minRange;
        scanMsg.range_max = maxRange;

        // Cast rays and collect ranges
        List<float> ranges = new List<float>();
        float angleStep = (maxAngle - minAngle) / numRays;

        for (int i = 0; i < numRays; i++)
        {
            float angle = minAngle + i * angleStep;
            float range = CastRayAtAngle(angle);
            ranges.Add(range);
        }

        scanMsg.ranges = ranges.ToArray();

        // Publish scan
        ros.Publish(scanTopic, scanMsg);
    }

    float CastRayAtAngle(float angleDegrees)
    {
        Vector3 direction = Quaternion.Euler(0, angleDegrees, 0) * transform.forward;
        RaycastHit hit;

        if (Physics.Raycast(transform.position, direction, out hit, maxRange))
        {
            return hit.distance;
        }
        else
        {
            // Return max range if no hit
            return maxRange;
        }
    }
}
```

### IMU Sensor Simulation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class UnityIMUSensor : MonoBehaviour
{
    [Header("IMU Configuration")]
    [SerializeField] private float updateRate = 100f;
    [SerializeField] private string imuTopic = "/imu/data";
    [SerializeField] private float noiseLevel = 0.01f;

    private ROSConnection ros;
    private Rigidbody rb;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        rb = GetComponentInParent<Rigidbody>();

        if (rb == null)
        {
            rb = GetComponent<Rigidbody>();
        }

        updateInterval = 1f / updateRate;
        lastUpdateTime = 0f;
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishIMUData();
            lastUpdateTime = Time.time;
        }
    }

    void PublishIMUData()
    {
        ImuMsg imuMsg = new ImuMsg();

        // Set header
        imuMsg.header = new std_msgs.HeaderMsg();
        imuMsg.header.stamp = new TimeMsg(0, (uint)(Time.time * 1e9));
        imuMsg.header.frame_id = transform.name;

        // Set orientation (from Unity rotation to ROS quaternion)
        Quaternion unityRotation = transform.rotation;
        QuaternionMsg orientation = new QuaternionMsg(
            unityRotation.x, unityRotation.y, unityRotation.z, unityRotation.w
        );
        imuMsg.orientation = orientation;

        // Set angular velocity (approximate from rotation change)
        imuMsg.angular_velocity = new Vector3Msg(
            AddNoise(rb.angularVelocity.x),
            AddNoise(rb.angularVelocity.y),
            AddNoise(rb.angularVelocity.z)
        );

        // Set linear acceleration (approximate from forces)
        Vector3 linearAcc = rb.velocity / Time.deltaTime;
        imuMsg.linear_acceleration = new Vector3Msg(
            AddNoise(linearAcc.x),
            AddNoise(linearAcc.y + 9.81f), // Account for gravity
            AddNoise(linearAcc.z)
        );

        // Publish IMU data
        ros.Publish(imuTopic, imuMsg);
    }

    float AddNoise(float value)
    {
        return value + Random.Range(-noiseLevel, noiseLevel);
    }
}
```

## Unity Perception Package

The Unity Perception package enables synthetic data generation for computer vision tasks:

### Semantic Segmentation

```csharp
using UnityEngine;
using Unity.Perception.GroundTruth;
using Unity.Perception.GroundTruth.DataModel;

public class SemanticSegmentationSetup : MonoBehaviour
{
    [Header("Semantic Segmentation")]
    [SerializeField] private bool enableSegmentation = true;
    [SerializeField] private Camera segmentationCamera;

    void Start()
    {
        if (enableSegmentation && segmentationCamera != null)
        {
            // Add semantic segmentation sensor
            var segmentationSensor = segmentationCamera.gameObject.AddComponent<SemanticSegmentationSensor>();

            // Configure the sensor
            segmentationSensor.Camera = segmentationCamera;
            segmentationSensor.Enabled = true;
        }
    }
}
```

### Bounding Box Labels

```csharp
using UnityEngine;
using Unity.Perception.GroundTruth;
using Unity.Perception.GroundTruth.Consumers;

public class BoundingBoxLabeler : MonoBehaviour
{
    [Header("Bounding Box Configuration")]
    [SerializeField] private string objectLabel = "object";
    [SerializeField] private int objectId = 0;

    void Start()
    {
        // Tag this object for bounding box detection
        var labeler = gameObject.AddComponent<Labeler>();
        labeler.label = objectLabel;
        labeler.id = objectId;
    }
}
```

## Advanced Unity Simulation Features

### Physics-Based Simulation

```csharp
using UnityEngine;

public class PhysicsRobot : MonoBehaviour
{
    [Header("Robot Physics")]
    [SerializeField] private float wheelRadius = 0.1f;
    [SerializeField] private float wheelSeparation = 0.4f;
    [SerializeField] private float maxMotorForce = 100f;

    [Header("Wheel Colliders")]
    [SerializeField] private WheelCollider leftWheel;
    [SerializeField] private WheelCollider rightWheel;

    [Header("Motor Control")]
    private float leftWheelMotor = 0f;
    private float rightWheelMotor = 0f;

    void FixedUpdate()
    {
        // Apply motor torques to wheels
        leftWheel.motorTorque = leftWheelMotor * maxMotorForce;
        rightWheel.motorTorque = rightWheelMotor * maxMotorForce;

        // Update wheel visual positions
        UpdateWheelVisuals();
    }

    void UpdateWheelVisuals()
    {
        // Update visual wheel positions based on physics simulation
        UpdateWheelVisual(leftWheel, GameObject.Find("LeftWheelVisual"));
        UpdateWheelVisual(rightWheel, GameObject.Find("RightWheelVisual"));
    }

    void UpdateWheelVisual(WheelCollider collider, GameObject wheelObject)
    {
        if (wheelObject != null)
        {
            Vector3 position;
            Quaternion rotation;
            collider.GetWorldPose(out position, out rotation);

            wheelObject.transform.position = position;
            wheelObject.transform.rotation = rotation;
        }
    }

    public void SetMotorCommands(float left, float right)
    {
        leftWheelMotor = left;
        rightWheelMotor = right;
    }
}
```

### Environment Generation

```csharp
using UnityEngine;
using System.Collections.Generic;

public class ProceduralEnvironment : MonoBehaviour
{
    [Header("Environment Generation")]
    [SerializeField] private GameObject[] obstaclePrefabs;
    [SerializeField] private int numObstacles = 10;
    [SerializeField] private Vector2 bounds = new Vector2(10f, 10f);
    [SerializeField] private float minSpacing = 2f;

    private List<Vector3> placedPositions = new List<Vector3>();

    void Start()
    {
        GenerateEnvironment();
    }

    void GenerateEnvironment()
    {
        for (int i = 0; i < numObstacles; i++)
        {
            Vector3 position = FindValidPosition();
            if (position != Vector3.zero)
            {
                GameObject obstacle = Instantiate(
                    obstaclePrefabs[Random.Range(0, obstaclePrefabs.Length)],
                    position,
                    Quaternion.Euler(0, Random.Range(0, 360), 0)
                );

                placedPositions.Add(position);
            }
        }
    }

    Vector3 FindValidPosition()
    {
        int attempts = 0;
        const int maxAttempts = 100;

        while (attempts < maxAttempts)
        {
            Vector3 candidate = new Vector3(
                Random.Range(-bounds.x, bounds.x),
                0.5f, // Height above ground
                Random.Range(-bounds.y, bounds.y)
            );

            // Check spacing from other obstacles
            bool valid = true;
            foreach (Vector3 pos in placedPositions)
            {
                if (Vector3.Distance(candidate, pos) < minSpacing)
                {
                    valid = false;
                    break;
                }
            }

            if (valid)
                return candidate;

            attempts++;
        }

        return Vector3.zero; // Failed to find valid position
    }
}
```

## Integration with ROS 2

### Unity ROS TCP Connector Setup

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class ROSIntegrationManager : MonoBehaviour
{
    [Header("ROS Connection")]
    [SerializeField] private string rosIP = "127.0.0.1";
    [SerializeField] private int rosPort = 10000;
    [SerializeField] private bool autoConnect = true;

    private ROSConnection rosConnection;

    void Start()
    {
        // Get or create ROS connection
        rosConnection = ROSConnection.GetOrCreateInstance();

        if (autoConnect)
        {
            rosConnection.Initialize(rosIP, rosPort);
        }
    }

    public void ConnectToROS(string ip, int port)
    {
        rosConnection.Initialize(ip, port);
    }

    public void DisconnectFromROS()
    {
        rosConnection.Disconnect();
    }
}
```

### Custom ROS Messages

```csharp
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using RosMessageTypes.Std;

// Custom message example
[System.Serializable]
public class RobotStatusMsg : Message
{
    public const string k_RosMessageName = "my_robot_msgs/RobotStatus";
    public override string RosMessageName => k_RosMessageName;

    public string robot_name;
    public bool is_active;
    public float battery_level;
    public int error_code;

    public RobotStatusMsg()
    {
        robot_name = "";
        is_active = false;
        battery_level = 0.0f;
        error_code = 0;
    }

    public RobotStatusMsg(string robot_name, bool is_active, float battery_level, int error_code)
    {
        this.robot_name = robot_name;
        this.is_active = is_active;
        this.battery_level = battery_level;
        this.error_code = error_code;
    }
}
```

## Unity Simulation Best Practices

### Performance Optimization

```csharp
using UnityEngine;

public class OptimizedSimulation : MonoBehaviour
{
    [Header("Performance Settings")]
    [SerializeField] private int targetFrameRate = 60;
    [SerializeField] private bool useFixedTimestep = true;
    [SerializeField] private float fixedTimestep = 0.02f; // 50 Hz

    void Start()
    {
        // Set target frame rate
        Application.targetFrameRate = targetFrameRate;

        // Configure time settings for consistent simulation
        if (useFixedTimestep)
        {
            Time.fixedDeltaTime = fixedTimestep;
        }

        // Optimize for simulation
        QualitySettings.vSyncCount = 0;
    }

    void Update()
    {
        // Simulation-specific optimizations
        OptimizeForSimulation();
    }

    void OptimizeForSimulation()
    {
        // Reduce rendering quality for better physics performance
        if (Time.timeScale < 1.0f)
        {
            QualitySettings.SetQualityLevel(0); // Lowest quality
        }
    }
}
```

### Multi-Robot Simulation

```csharp
using UnityEngine;
using System.Collections.Generic;

public class MultiRobotManager : MonoBehaviour
{
    [Header("Multi-Robot Setup")]
    [SerializeField] private GameObject robotPrefab;
    [SerializeField] private int numRobots = 5;
    [SerializeField] private Vector2 spawnBounds = new Vector2(10f, 10f);

    private List<GameObject> robots = new List<GameObject>();

    void Start()
    {
        SpawnRobots();
    }

    void SpawnRobots()
    {
        for (int i = 0; i < numRobots; i++)
        {
            Vector3 spawnPosition = new Vector3(
                Random.Range(-spawnBounds.x, spawnBounds.x),
                0.1f, // Slightly above ground
                Random.Range(-spawnBounds.y, spawnBounds.y)
            );

            GameObject robot = Instantiate(robotPrefab, spawnPosition, Quaternion.identity);
            robot.name = $"Robot_{i:D3}";

            // Assign unique ROS topics for each robot
            RobotController controller = robot.GetComponent<RobotController>();
            if (controller != null)
            {
                controller.SetRobotNamespace($"robot_{i}");
            }

            robots.Add(robot);
        }
    }
}
```

## Unity vs Gazebo Comparison

| Feature | Unity | Gazebo |
|---------|-------|--------|
| Graphics Quality | Photorealistic | Basic visualization |
| Physics Accuracy | Good | Excellent |
| Sensor Simulation | Excellent for vision | Comprehensive |
| ROS Integration | Good (TCP) | Excellent (native) |
| Performance | High visual fidelity cost | Optimized for physics |
| Learning Curve | Moderate (Unity knowledge) | Moderate (Gazebo-specific) |
| Use Cases | Vision, ML, HMI | Control, navigation, physics |

## Practical Example: Warehouse Navigation Simulation

Let's create a complete example that combines multiple Unity features:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Nav;
using RosMessageTypes.Geometry;

public class WarehouseNavigation : MonoBehaviour
{
    [Header("Navigation Configuration")]
    [SerializeField] private string moveBaseTopic = "/move_base_simple/goal";
    [SerializeField] private string cmdVelTopic = "/cmd_vel";
    [SerializeField] private Transform goalMarker;
    [SerializeField] private float navigationThreshold = 0.5f;

    private ROSConnection ros;
    private Transform robotTransform;
    private bool hasGoal = false;
    private Vector3 currentGoal;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        robotTransform = transform;

        // Subscribe to navigation goals
        ros.Subscribe<PoseStampedMsg>(moveBaseTopic, OnNavigationGoal);

        // Initialize goal marker
        if (goalMarker != null)
        {
            goalMarker.gameObject.SetActive(false);
        }
    }

    void OnNavigationGoal(PoseStampedMsg goalMsg)
    {
        // Convert ROS pose to Unity position
        currentGoal = new Vector3(
            (float)goalMsg.pose.position.x,
            (float)goalMsg.pose.position.y,
            (float)goalMsg.pose.position.z
        );

        hasGoal = true;

        // Update goal marker
        if (goalMarker != null)
        {
            goalMarker.position = currentGoal;
            goalMarker.gameObject.SetActive(true);
        }

        Debug.Log($"Received navigation goal: {currentGoal}");
    }

    void Update()
    {
        if (hasGoal)
        {
            NavigateToGoal();
        }
    }

    void NavigateToGoal()
    {
        Vector3 direction = currentGoal - transform.position;
        float distance = direction.magnitude;

        if (distance > navigationThreshold)
        {
            // Simple navigation logic (in practice, use path planning)
            direction.Normalize();

            // Publish velocity command
            TwistMsg cmd = new TwistMsg();
            cmd.linear = new Vector3Msg(direction.x * 0.5, 0, 0); // Move forward
            cmd.angular = new Vector3Msg(0, 0, -direction.z * 0.2); // Turn if needed

            ros.Publish(cmdVelTopic, cmd);
        }
        else
        {
            // Reached goal
            hasGoal = false;

            // Publish zero velocity
            TwistMsg stopCmd = new TwistMsg();
            ros.Publish(cmdVelTopic, stopCmd);

            Debug.Log("Reached navigation goal!");
        }
    }
}
```

## Key Takeaways

- Unity provides photorealistic rendering capabilities for advanced perception tasks
- Unity Robotics Hub enables seamless integration with ROS ecosystems
- Sensor simulation in Unity can generate synthetic data for AI training
- Unity's physics engine supports realistic robot interactions
- Performance optimization is crucial for real-time simulation
- Unity complements Gazebo by excelling in visual perception tasks

## Next Steps

In the next chapter, we'll explore simulation-to-reality transfer techniques, learning how to bridge the gap between simulated and real-world robot performance.