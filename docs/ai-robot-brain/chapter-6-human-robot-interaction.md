---
sidebar_position: 6
title: Human-Robot Interaction
---

# Human-Robot Interaction

In this chapter, we'll explore how AI robot brains enable natural, intuitive, and effective interaction between humans and robots. Human-robot interaction (HRI) is crucial for creating robots that can work alongside humans, understand human intentions, and respond appropriately to human commands and social cues.

## Understanding Human-Robot Interaction

Human-robot interaction encompasses several key aspects:

### 1. Communication Modalities
- **Verbal communication**: Speech recognition and natural language processing
- **Non-verbal communication**: Gestures, facial expressions, body language
- **Tactile communication**: Physical interaction and haptic feedback
- **Visual communication**: Displays, lights, and visual indicators

### 2. Social Intelligence
- **Social norms awareness**: Understanding and following social conventions
- **Emotional intelligence**: Recognizing and responding to human emotions
- **Context awareness**: Understanding the social and environmental context
- **Personalization**: Adapting to individual human preferences and capabilities

### 3. Collaborative Behaviors
- **Joint attention**: Focusing on the same object or task
- **Turn-taking**: Coordinating actions and communication
- **Role negotiation**: Determining who does what in collaborative tasks
- **Trust building**: Establishing and maintaining trust over time

## Natural Language Processing for HRI

### Speech Recognition and Understanding

```python
import speech_recognition as sr
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Set up ambient noise threshold
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def listen_and_recognize(self, timeout: int = 5) -> str:
        """Listen to speech and return recognized text"""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)

            # Recognize speech using Google's API
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""

class IntentClassifier(nn.Module):
    def __init__(self, num_intents: int):
        super(IntentClassifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_intents)  # BERT hidden size is 768

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class NaturalLanguageProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.intent_model = None  # Will be loaded or trained
        self.intents = {
            0: 'greeting',
            1: 'navigation',
            2: 'manipulation',
            3: 'information_request',
            4: 'help_request',
            5: 'goodbye'
        }

    def preprocess_text(self, text: str) -> dict:
        """Preprocess text for BERT model"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=128,
            padding='max_length',
            truncation=True
        )
        return inputs

    def classify_intent(self, text: str) -> str:
        """Classify the intent of the given text"""
        if self.intent_model is None:
            # Fallback to simple keyword matching if no trained model
            return self._keyword_based_intent(text)

        inputs = self.preprocess_text(text)
        with torch.no_grad():
            outputs = self.intent_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        return self.intents.get(predicted_class, 'unknown')

    def _keyword_based_intent(self, text: str) -> str:
        """Simple keyword-based intent classification"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting'
        elif any(word in text_lower for word in ['go to', 'navigate', 'move to', 'take me to']):
            return 'navigation'
        elif any(word in text_lower for word in ['pick up', 'grasp', 'take', 'get']):
            return 'manipulation'
        elif any(word in text_lower for word in ['what', 'how', 'when', 'where', 'who', 'information']):
            return 'information_request'
        elif any(word in text_lower for word in ['help', 'assist', 'can you', 'could you']):
            return 'help_request'
        elif any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
            return 'goodbye'
        else:
            return 'unknown'

    def extract_entities(self, text: str) -> dict:
        """Extract named entities from text"""
        entities = {
            'locations': [],
            'objects': [],
            'people': [],
            'times': []
        }

        # Simple rule-based entity extraction
        text_lower = text.lower()
        words = text_lower.split()

        # Common locations
        common_locations = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'dining room']
        entities['locations'] = [word for word in words if word in common_locations]

        # Common objects
        common_objects = ['cup', 'book', 'phone', 'keys', 'water', 'food', 'bottle']
        entities['objects'] = [word for word in words if word in common_objects]

        return entities
```

### Dialogue Management System

```python
from typing import Dict, List, Optional, Tuple
import re

class DialogueState:
    def __init__(self):
        self.current_intent = None
        self.entities = {}
        self.context = {}
        self.turn_count = 0
        self.user_profile = {}
        self.task_stack = []

class DialogueManager:
    def __init__(self):
        self.state = DialogueState()
        self.nlp_processor = NaturalLanguageProcessor()
        self.response_templates = self._load_response_templates()

    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load response templates for different intents"""
        return {
            'greeting': [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I help you?"
            ],
            'navigation': [
                "I can help you navigate to {location}.",
                "Sure, I'll take you to {location}.",
                "Okay, heading to {location} now."
            ],
            'manipulation': [
                "I'll help you with that {object}.",
                "Sure, I can get the {object} for you.",
                "Okay, I'll pick up the {object}."
            ],
            'information_request': [
                "I can provide information about {topic}.",
                "Let me find that information for you.",
                "Here's what I know about {topic}."
            ],
            'help_request': [
                "I'm here to help. What do you need?",
                "Sure, I can assist you with that.",
                "How can I be of service?"
            ],
            'goodbye': [
                "Goodbye! Have a great day!",
                "See you later!",
                "Farewell! Feel free to call me if you need anything."
            ]
        }

    def process_input(self, user_input: str) -> str:
        """Process user input and generate appropriate response"""
        self.state.turn_count += 1

        # Classify intent
        intent = self.nlp_processor.classify_intent(user_input)
        self.state.current_intent = intent

        # Extract entities
        entities = self.nlp_processor.extract_entities(user_input)
        self.state.entities.update(entities)

        # Generate response based on intent and context
        response = self._generate_response(intent, entities, user_input)

        return response

    def _generate_response(self, intent: str, entities: Dict, user_input: str) -> str:
        """Generate appropriate response based on intent and entities"""
        if intent == 'navigation' and entities.get('locations'):
            location = entities['locations'][0]
            template = self.response_templates['navigation'][0]
            return template.format(location=location)
        elif intent == 'manipulation' and entities.get('objects'):
            obj = entities['objects'][0]
            template = self.response_templates['manipulation'][0]
            return template.format(object=obj)
        elif intent in self.response_templates:
            template = self.response_templates[intent][0]
            return template.format(topic=entities.get('objects', ['that'])[0] if entities.get('objects') else 'this')
        else:
            return "I'm not sure I understand. Could you please rephrase that?"

    def handle_context_switch(self, new_context: str):
        """Handle switching to a new context or task"""
        if self.state.current_intent:
            # Save current task
            self.state.task_stack.append({
                'intent': self.state.current_intent,
                'entities': self.state.entities.copy(),
                'context': self.state.context.copy()
            })

        # Reset for new context
        self.state.current_intent = None
        self.state.entities = {}
        self.state.context = {'active_task': new_context}

    def resume_previous_task(self):
        """Resume a previously interrupted task"""
        if self.state.task_stack:
            previous_task = self.state.task_stack.pop()
            self.state.current_intent = previous_task['intent']
            self.state.entities.update(previous_task['entities'])
            self.state.context.update(previous_task['context'])
            return True
        return False

    def update_user_profile(self, user_id: str, preferences: Dict):
        """Update user profile with preferences"""
        if user_id not in self.state.user_profile:
            self.state.user_profile[user_id] = {}
        self.state.user_profile[user_id].update(preferences)
```

## Gesture Recognition and Understanding

### Computer Vision for Gesture Recognition

```python
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.gesture_templates = self._load_gesture_templates()

    def _load_gesture_templates(self) -> Dict[str, List[List[float]]]:
        """Load predefined gesture templates"""
        # Each template is a list of normalized landmark positions
        return {
            'pointing': [
                [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -0.5, 0.0],  # Thumb
                [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -0.8, 0.0],  # Index finger extended
                [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.3, 0.0],    # Other fingers curled
                [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.3, 0.0],
                [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.3, 0.0]
            ],
            'open_palm': [
                [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -0.8, 0.0],  # All fingers extended
                [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -0.8, 0.0],
                [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -0.8, 0.0],
                [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -0.8, 0.0],
                [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -0.8, 0.0]
            ],
            'stop': [
                [0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, -0.3, 0.0],  # Thumb
                [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -0.8, 0.0],  # Index finger extended
                [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -0.8, 0.0],  # Middle finger extended
                [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.3, 0.0],    # Ring finger curled
                [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.3, 0.0]     # Pinky curled
            ]
        }

    def detect_gestures(self, image: np.ndarray) -> List[Dict]:
        """Detect hand gestures in the image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        gestures = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark positions
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

                # Classify gesture
                gesture_type = self._classify_gesture(landmarks)
                gesture_confidence = self._calculate_gesture_confidence(landmarks, gesture_type)

                gesture_info = {
                    'type': gesture_type,
                    'confidence': gesture_confidence,
                    'landmarks': landmarks,
                    'center': self._calculate_hand_center(landmarks)
                }

                gestures.append(gesture_info)

                # Draw landmarks on image
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        return gestures

    def _classify_gesture(self, landmarks: List[List[float]]) -> str:
        """Classify gesture based on landmark positions"""
        # Simple distance-based gesture classification
        # Compare with template gestures using distance metrics

        # Calculate finger positions relative to palm
        palm_center = landmarks[0]  # Wrist as reference
        finger_tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]  # Finger tips

        # Simple heuristic-based classification
        extended_fingers = 0
        for tip in finger_tips:
            # Check if finger is extended (distance from palm)
            distance = np.sqrt((tip[0] - palm_center[0])**2 + (tip[1] - palm_center[1])**2)
            if distance > 0.1:  # Threshold for extended finger
                extended_fingers += 1

        if extended_fingers == 5:
            return 'open_palm'
        elif extended_fingers == 2 and self._is_pointing_gesture(landmarks):
            return 'pointing'
        elif extended_fingers == 2 and self._is_stop_gesture(landmarks):
            return 'stop'
        else:
            return 'unknown'

    def _is_pointing_gesture(self, landmarks: List[List[float]]) -> bool:
        """Check if landmarks form a pointing gesture"""
        # Index finger extended, others curled
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        palm_center = landmarks[0]

        # Index finger should be extended while middle finger is curled
        index_distance = np.sqrt((index_tip[0] - palm_center[0])**2 + (index_tip[1] - palm_center[1])**2)
        middle_distance = np.sqrt((middle_tip[0] - palm_center[0])**2 + (middle_tip[1] - palm_center[1])**2)

        return index_distance > 0.15 and middle_distance < 0.1

    def _is_stop_gesture(self, landmarks: List[List[float]]) -> bool:
        """Check if landmarks form a stop gesture"""
        # Palm facing forward with fingers extended
        # For simplicity, we'll check if palm is open with 2-3 fingers extended
        extended_count = 0
        palm_center = landmarks[0]

        for i in [8, 12, 16]:  # Index, middle, ring finger tips
            tip = landmarks[i]
            distance = np.sqrt((tip[0] - palm_center[0])**2 + (tip[1] - palm_center[1])**2)
            if distance > 0.1:
                extended_count += 1

        return extended_count >= 2

    def _calculate_gesture_confidence(self, landmarks: List[List[float]], gesture_type: str) -> float:
        """Calculate confidence of gesture classification"""
        # For now, return a simple confidence based on landmark visibility
        return 0.9  # High confidence if landmarks are detected

    def _calculate_hand_center(self, landmarks: List[List[float]]) -> Tuple[float, float]:
        """Calculate approximate center of the hand"""
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        return (np.mean(x_coords), np.mean(y_coords))

class BodyPoseRecognizer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def detect_pose(self, image: np.ndarray) -> Dict:
        """Detect body pose in the image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

            pose_info = {
                'landmarks': landmarks,
                'center': self._calculate_body_center(landmarks),
                'orientation': self._calculate_orientation(landmarks)
            }

            return pose_info

        return {'landmarks': [], 'center': (0, 0), 'orientation': 0}

    def _calculate_body_center(self, landmarks: List[List[float]]) -> Tuple[float, float]:
        """Calculate center of the body"""
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        return (np.mean(x_coords), np.mean(y_coords))

    def _calculate_orientation(self, landmarks: List[List[float]]) -> float:
        """Calculate body orientation based on shoulder positions"""
        if len(landmarks) >= 12:  # Ensure we have shoulder landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            # Calculate angle between shoulders
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]
            return np.arctan2(dy, dx)
        return 0.0
```

## Social Signal Processing

### Emotion Recognition

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import dlib

class EmotionRecognizer:
    def __init__(self):
        # Load pre-trained emotion recognition model
        # Note: In practice, you would load a model trained on facial expressions
        self.emotion_model = None  # Placeholder - would load actual model
        self.face_detector = dlib.get_frontal_face_detector()
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def detect_emotions(self, image: np.ndarray) -> List[Dict]:
        """Detect emotions from facial expressions in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)

        emotions = []

        for face in faces:
            # Extract face region
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_roi = gray[y:y+h, x:x+w]

            # Preprocess face for emotion recognition
            face_resized = cv2.resize(face_roi, (48, 48))
            face_normalized = face_resized / 255.0
            face_array = img_to_array(face_normalized)
            face_array = np.expand_dims(face_array, axis=0)
            face_array = np.expand_dims(face_array, axis=-1)

            # Predict emotion (placeholder - would use actual model)
            # emotions_probs = self.emotion_model.predict(face_array)
            # emotion_idx = np.argmax(emotions_probs)
            # emotion_label = self.emotion_labels[emotion_idx]
            # confidence = emotions_probs[0][emotion_idx]

            # For demonstration, return random emotion
            import random
            emotion_label = random.choice(self.emotion_labels)
            confidence = random.uniform(0.6, 0.9)

            emotion_info = {
                'emotion': emotion_label,
                'confidence': confidence,
                'bbox': (x, y, w, h),
                'region': face_roi
            }

            emotions.append(emotion_info)

        return emotions

class SocialContextAnalyzer:
    def __init__(self):
        self.proxemics_zones = {
            'intimate': 0.5,      # 0-0.5m
            'personal': 1.2,      # 0.5-1.2m
            'social': 3.7,        # 1.2-3.7m
            'public': float('inf') # 3.7m+
        }

    def analyze_social_context(self, robot_pos: Tuple[float, float, float],
                              human_pos: Tuple[float, float, float]) -> Dict:
        """Analyze social context based on spatial relationships"""
        distance = np.sqrt(sum((a - b)**2 for a, b in zip(robot_pos, human_pos)))

        # Determine proxemics zone
        zone = 'public'
        for zone_name, zone_limit in self.proxemics_zones.items():
            if distance <= zone_limit:
                zone = zone_name
                break

        context_info = {
            'distance': distance,
            'proxemics_zone': zone,
            'social_comfort': self._calculate_social_comfort(distance),
            'recommended_behavior': self._get_recommended_behavior(zone)
        }

        return context_info

    def _calculate_social_comfort(self, distance: float) -> float:
        """Calculate social comfort level based on distance"""
        # Comfortable distance is typically personal space (0.5-1.2m)
        if 0.5 <= distance <= 1.2:
            return 1.0  # Maximum comfort
        elif distance < 0.5:
            return 0.2  # Too close
        elif distance > 3.7:
            return 0.3  # Too far for interaction
        else:
            # Linear interpolation
            if distance < 1.2:
                return 0.2 + 0.8 * (distance - 0.5) / 0.7
            else:
                return 0.3 + 0.7 * (3.7 - distance) / 2.5

    def _get_recommended_behavior(self, zone: str) -> str:
        """Get recommended behavior based on proxemics zone"""
        behavior_map = {
            'intimate': 'Maintain respectful distance, avoid sudden movements',
            'personal': 'Appropriate for close interaction, maintain eye contact',
            'social': 'Good for general conversation, respectful interaction',
            'public': 'Increase volume, use gestures for communication'
        }
        return behavior_map.get(zone, 'Unknown zone')
```

## Collaborative Task Management

### Joint Attention and Task Coordination

```python
from typing import List, Dict, Tuple, Optional
import numpy as np

class JointAttentionSystem:
    def __init__(self):
        self.attention_targets = {}
        self.human_attention = None
        self.robot_attention = None
        self.shared_attention = None

    def update_human_attention(self, gaze_target: Tuple[float, float, float],
                              attention_confidence: float = 1.0):
        """Update human attention based on gaze or pointing"""
        self.human_attention = {
            'target': gaze_target,
            'confidence': attention_confidence,
            'timestamp': time.time()
        }

    def update_robot_attention(self, focus_target: Tuple[float, float, float]):
        """Update robot attention focus"""
        self.robot_attention = {
            'target': focus_target,
            'timestamp': time.time()
        }

    def detect_joint_attention(self, threshold: float = 0.1) -> bool:
        """Detect if human and robot are attending to the same object"""
        if self.human_attention and self.robot_attention:
            human_target = np.array(self.human_attention['target'])
            robot_target = np.array(self.robot_attention['target'])

            distance = np.linalg.norm(human_target - robot_target)
            return distance < threshold
        return False

    def get_shared_attention_object(self) -> Optional[Dict]:
        """Get the object of shared attention"""
        if self.detect_joint_attention():
            # In practice, this would involve object recognition
            # to identify what both are looking at
            return {
                'position': self.human_attention['target'],
                'type': 'shared_object',  # Would be determined by object recognition
                'confidence': min(self.human_attention['confidence'], 0.8)
            }
        return None

class CollaborativeTaskManager:
    def __init__(self):
        self.active_tasks = []
        self.task_assignments = {}
        self.collaboration_state = 'idle'
        self.turn_manager = TurnManager()

    def initiate_collaboration(self, task_description: str) -> bool:
        """Initiate a collaborative task with human"""
        task = {
            'id': len(self.active_tasks),
            'description': task_description,
            'status': 'initiated',
            'participants': ['human', 'robot'],
            'subtasks': self._decompose_task(task_description),
            'current_subtask': 0
        }

        self.active_tasks.append(task)
        self.collaboration_state = 'active'

        # Assign initial subtasks
        self._assign_subtasks(task)

        return True

    def _decompose_task(self, task_description: str) -> List[Dict]:
        """Decompose high-level task into subtasks"""
        # Simple decomposition based on keywords
        if 'assemble' in task_description.lower():
            return [
                {'id': 0, 'action': 'locate_parts', 'assigned_to': 'robot'},
                {'id': 1, 'action': 'hand_over_part', 'assigned_to': 'robot'},
                {'id': 2, 'action': 'assemble', 'assigned_to': 'human'},
                {'id': 3, 'action': 'inspect', 'assigned_to': 'robot'}
            ]
        elif 'cook' in task_description.lower():
            return [
                {'id': 0, 'action': 'find_ingredients', 'assigned_to': 'robot'},
                {'id': 1, 'action': 'prepare_ingredients', 'assigned_to': 'human'},
                {'id': 2, 'action': 'cook', 'assigned_to': 'human'},
                {'id': 3, 'action': 'serve', 'assigned_to': 'robot'}
            ]
        else:
            # Default decomposition
            return [
                {'id': 0, 'action': 'understand_task', 'assigned_to': 'both'},
                {'id': 1, 'action': 'plan_execution', 'assigned_to': 'both'},
                {'id': 2, 'action': 'execute_task', 'assigned_to': 'both'},
                {'id': 3, 'action': 'verify_completion', 'assigned_to': 'both'}
            ]

    def _assign_subtasks(self, task: Dict):
        """Assign subtasks to human and robot based on capabilities"""
        for subtask in task['subtasks']:
            if subtask['assigned_to'] == 'both':
                # Use turn manager to alternate
                subtask['assigned_to'] = self.turn_manager.get_next_actor()
            elif subtask['assigned_to'] == 'robot':
                # Robot-specific tasks
                pass
            elif subtask['assigned_to'] == 'human':
                # Human-specific tasks
                pass

    def update_task_progress(self, task_id: int, subtask_id: int,
                           success: bool = True, feedback: str = ""):
        """Update progress of collaborative task"""
        for task in self.active_tasks:
            if task['id'] == task_id:
                if success:
                    task['subtasks'][subtask_id]['status'] = 'completed'
                    task['current_subtask'] = subtask_id + 1

                    # Move to next subtask
                    if task['current_subtask'] < len(task['subtasks']):
                        next_subtask = task['subtasks'][task['current_subtask']]
                        self._assign_subtasks_to_available(task, next_subtask)
                    else:
                        task['status'] = 'completed'
                        self.collaboration_state = 'completed'
                else:
                    task['subtasks'][subtask_id]['status'] = 'failed'
                    task['status'] = 'failed'
                    self.collaboration_state = 'failed'

                # Log feedback
                task['subtasks'][subtask_id]['feedback'] = feedback
                break

    def _assign_subtasks_to_available(self, task: Dict, subtask: Dict):
        """Assign subtasks based on availability and capability"""
        # In a real system, this would consider current workload,
        # capabilities, and preferences
        pass

    def get_collaboration_status(self) -> Dict:
        """Get current status of collaboration"""
        return {
            'state': self.collaboration_state,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len([t for t in self.active_tasks if t['status'] == 'completed']),
            'current_task': self.active_tasks[-1] if self.active_tasks else None
        }

class TurnManager:
    def __init__(self):
        self.current_actor = 'human'  # Start with human turn
        self.turn_history = []

    def get_next_actor(self) -> str:
        """Get the next actor in turn sequence"""
        next_actor = 'robot' if self.current_actor == 'human' else 'human'
        self.turn_history.append({
            'actor': self.current_actor,
            'timestamp': time.time()
        })
        self.current_actor = next_actor
        return next_actor

    def request_turn(self, requesting_actor: str) -> bool:
        """Request turn from current actor"""
        if requesting_actor == self.current_actor:
            return True
        else:
            # In a real system, this might involve negotiation
            # For now, we'll allow the request
            self.current_actor = requesting_actor
            return True
```

## Trust and Personalization

### Trust Modeling

```python
import numpy as np
from typing import Dict, List, Tuple
import time

class TrustModel:
    def __init__(self):
        self.trust_scores = {}
        self.interaction_history = []
        self.trust_decay_rate = 0.95  # Trust decays over time
        self.competence_weight = 0.6
        self.reliability_weight = 0.3
        self.benevolence_weight = 0.1

    def update_trust(self, user_id: str, interaction_outcome: Dict):
        """Update trust based on interaction outcome"""
        if user_id not in self.trust_scores:
            self.trust_scores[user_id] = {
                'competence': 0.5,
                'reliability': 0.5,
                'benevolence': 0.5,
                'overall': 0.5,
                'last_interaction': time.time()
            }

        current_trust = self.trust_scores[user_id]

        # Apply time decay
        time_since_interaction = time.time() - current_trust['last_interaction']
        decay_factor = self.trust_decay_rate ** (time_since_interaction / 3600)  # Decay per hour
        current_trust['competence'] *= decay_factor
        current_trust['reliability'] *= decay_factor
        current_trust['benevolence'] *= decay_factor

        # Update based on interaction outcome
        if 'success' in interaction_outcome:
            success = interaction_outcome['success']
            if success:
                current_trust['competence'] = min(1.0, current_trust['competence'] + 0.1)
                current_trust['reliability'] = min(1.0, current_trust['reliability'] + 0.05)
            else:
                current_trust['competence'] = max(0.0, current_trust['competence'] - 0.1)
                current_trust['reliability'] = max(0.0, current_trust['reliability'] - 0.05)

        if 'safety' in interaction_outcome:
            safety_rating = interaction_outcome['safety']  # 0-1 scale
            current_trust['benevolence'] = (
                current_trust['benevolence'] * 0.8 + safety_rating * 0.2
            )

        # Calculate overall trust
        current_trust['overall'] = (
            self.competence_weight * current_trust['competence'] +
            self.reliability_weight * current_trust['reliability'] +
            self.benevolence_weight * current_trust['benevolence']
        )

        current_trust['last_interaction'] = time.time()

        # Store interaction in history
        self.interaction_history.append({
            'user_id': user_id,
            'outcome': interaction_outcome,
            'timestamp': time.time(),
            'trust_after': current_trust.copy()
        })

    def get_trust_level(self, user_id: str) -> float:
        """Get current trust level for user"""
        if user_id in self.trust_scores:
            return self.trust_scores[user_id]['overall']
        return 0.5  # Default neutral trust

    def get_trust_breakdown(self, user_id: str) -> Dict:
        """Get detailed trust breakdown"""
        if user_id in self.trust_scores:
            return self.trust_scores[user_id]
        return {
            'competence': 0.5,
            'reliability': 0.5,
            'benevolence': 0.5,
            'overall': 0.5
        }

    def adjust_behavior_for_trust(self, user_id: str) -> Dict:
        """Adjust robot behavior based on trust level"""
        trust_level = self.get_trust_level(user_id)

        behavior_adjustment = {
            'communication_style': 'direct' if trust_level > 0.7 else 'cautious',
            'autonomy_level': 'high' if trust_level > 0.8 else 'medium' if trust_level > 0.5 else 'low',
            'proactivity': 'high' if trust_level > 0.7 else 'low',
            'verification_frequency': 'low' if trust_level > 0.7 else 'high',
            'safety_margin': 0.1 if trust_level > 0.7 else 0.3
        }

        return behavior_adjustment
```

### Personalization System

```python
class PersonalizationSystem:
    def __init__(self):
        self.user_profiles = {}
        self.preference_learner = PreferenceLearner()
        self.adaptation_engine = AdaptationEngine()

    def update_user_preferences(self, user_id: str, interaction_data: Dict):
        """Update user preferences based on interaction"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'preferences': {},
                'interaction_history': [],
                'communication_style': 'neutral',
                'pace_preference': 'medium',
                'feedback_preference': 'detailed'
            }

        profile = self.user_profiles[user_id]
        profile['interaction_history'].append(interaction_data)

        # Learn preferences from interaction
        new_preferences = self.preference_learner.analyze_interaction(
            interaction_data, profile
        )

        # Update profile with new preferences
        profile['preferences'].update(new_preferences)

        # Adapt robot behavior
        self.adaptation_engine.adapt_to_preferences(
            user_id, profile['preferences']
        )

    def get_personalized_response(self, user_id: str, base_response: str) -> str:
        """Get personalized response based on user profile"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            return self.adaptation_engine.personalize_response(
                base_response, profile
            )
        return base_response

    def get_personalized_behavior(self, user_id: str) -> Dict:
        """Get personalized behavior parameters"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            return self.adaptation_engine.get_adapted_behavior(profile)
        return {}

class PreferenceLearner:
    def analyze_interaction(self, interaction_data: Dict, profile: Dict) -> Dict:
        """Analyze interaction to learn user preferences"""
        preferences = {}

        # Analyze response time preferences
        if 'response_time' in interaction_data:
            if interaction_data['response_time'] < 2.0:
                preferences['pace_preference'] = 'fast'
            elif interaction_data['response_time'] > 5.0:
                preferences['pace_preference'] = 'slow'

        # Analyze feedback preferences
        if 'feedback_helpful' in interaction_data:
            if interaction_data['feedback_helpful']:
                preferences['feedback_preference'] = 'detailed'
            else:
                preferences['feedback_preference'] = 'concise'

        # Analyze communication style
        if 'communication_style' in interaction_data:
            preferences['communication_style'] = interaction_data['communication_style']

        # Analyze task preferences
        if 'task_preference' in interaction_data:
            if 'task_preference' not in preferences:
                preferences['task_preference'] = []
            preferences['task_preference'].append(interaction_data['task_preference'])

        return preferences

class AdaptationEngine:
    def __init__(self):
        self.behavior_templates = {
            'fast_paced': {
                'response_speed': 'quick',
                'verbosity': 'concise',
                'proactivity': 'high'
            },
            'slow_paced': {
                'response_speed': 'deliberate',
                'verbosity': 'detailed',
                'proactivity': 'low'
            },
            'detailed_feedback': {
                'explanation_level': 'comprehensive',
                'error_explanation': 'thorough',
                'progress_updates': 'frequent'
            },
            'concise_feedback': {
                'explanation_level': 'minimal',
                'error_explanation': 'brief',
                'progress_updates': 'occasional'
            }
        }

    def adapt_to_preferences(self, user_id: str, preferences: Dict):
        """Adapt robot behavior to user preferences"""
        # In a real system, this would update various robot parameters
        # such as speech rate, gesture frequency, response style, etc.
        pass

    def personalize_response(self, base_response: str, profile: Dict) -> str:
        """Personalize a response based on user profile"""
        preferences = profile.get('preferences', {})

        if preferences.get('communication_style') == 'formal':
            return "Certainly, " + base_response.lower()
        elif preferences.get('communication_style') == 'casual':
            return "Sure thing! " + base_response
        else:
            return base_response

    def get_adapted_behavior(self, profile: Dict) -> Dict:
        """Get behavior parameters adapted to user"""
        preferences = profile.get('preferences', {})

        behavior = {
            'speech_rate': 1.0,  # Normal rate
            'gesture_frequency': 0.5,  # Medium frequency
            'interaction_distance': 1.0,  # Normal distance
            'proactivity_level': 0.5  # Medium proactivity
        }

        if preferences.get('pace_preference') == 'fast':
            behavior['speech_rate'] = 1.2
            behavior['proactivity_level'] = 0.8
        elif preferences.get('pace_preference') == 'slow':
            behavior['speech_rate'] = 0.8
            behavior['proactivity_level'] = 0.2

        if preferences.get('communication_style') == 'expressive':
            behavior['gesture_frequency'] = 0.8

        return behavior
```

## NVIDIA Isaac HRI Integration

### Isaac ROS HRI Components

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped, PoseStamped
from audio_common_msgs.msg import AudioData
import numpy as np

class IsaacHRINode(Node):
    def __init__(self):
        super().__init__('isaac_hri_node')

        # Publishers
        self.speech_pub = self.create_publisher(String, '/speech_output', 10)
        self.gesture_pub = self.create_publisher(String, '/gesture_commands', 10)
        self.face_display_pub = self.create_publisher(String, '/face_display', 10)
        self.audio_feedback_pub = self.create_publisher(AudioData, '/audio_feedback', 10)

        # Subscribers
        self.speech_sub = self.create_subscription(
            String, '/speech_input', self.speech_callback, 10)
        self.gesture_sub = self.create_subscription(
            String, '/gesture_input', self.gesture_callback, 10)
        self.face_detection_sub = self.create_subscription(
            Image, '/face_detection/image', self.face_detection_callback, 10)
        self.gaze_sub = self.create_subscription(
            PointStamped, '/gaze_direction', self.gaze_callback, 10)

        # HRI components
        self.dialogue_manager = DialogueManager()
        self.gesture_recognizer = HandGestureRecognizer()
        self.trust_model = TrustModel()
        self.personalization_system = PersonalizationSystem()

        # User tracking
        self.current_user = None
        self.user_interaction_count = {}

        # HRI parameters
        self.declare_parameter('hri_enabled', True)
        self.declare_parameter('trust_threshold', 0.6)
        self.hri_enabled = self.get_parameter('hri_enabled').value
        self.trust_threshold = self.get_parameter('trust_threshold').value

    def speech_callback(self, msg: String):
        """Handle incoming speech input"""
        if not self.hri_enabled:
            return

        user_input = msg.data
        self.get_logger().info(f"Received speech: {user_input}")

        # Process with dialogue manager
        response = self.dialogue_manager.process_input(user_input)

        # Publish response
        response_msg = String()
        response_msg.data = response
        self.speech_pub.publish(response_msg)

        # Update trust based on interaction
        self._update_user_trust(user_input, response)

    def gesture_callback(self, msg: String):
        """Handle incoming gesture input"""
        if not self.hri_enabled:
            return

        gesture_data = msg.data
        self.get_logger().info(f"Received gesture: {gesture_data}")

        # Process gesture and update dialogue state
        self._process_gesture(gesture_data)

    def face_detection_callback(self, msg: Image):
        """Handle face detection input"""
        if not self.hri_enabled:
            return

        # Convert ROS Image to OpenCV format
        # This is a simplified representation
        # In practice, you'd use cv_bridge to convert the image
        # image_cv = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        # Detect faces and update user tracking
        self._update_user_tracking()

    def gaze_callback(self, msg: PointStamped):
        """Handle gaze direction input"""
        if not self.hri_enabled:
            return

        gaze_point = (msg.point.x, msg.point.y, msg.point.z)
        self.get_logger().info(f"Gaze direction: {gaze_point}")

        # Update joint attention system
        self._update_joint_attention(gaze_point)

    def _update_user_trust(self, user_input: str, response: str):
        """Update trust model based on interaction"""
        if self.current_user:
            # Simple success metric based on response length and positivity
            success = len(response) > 10  # Arbitrary threshold
            interaction_outcome = {
                'success': success,
                'safety': 1.0 if 'please' in user_input.lower() or 'thank' in user_input.lower() else 0.8
            }

            self.trust_model.update_trust(self.current_user, interaction_outcome)

            # Adjust behavior based on trust level
            trust_level = self.trust_model.get_trust_level(self.current_user)
            if trust_level < self.trust_threshold:
                self.get_logger().warn(f"Low trust level for user {self.current_user}")

    def _process_gesture(self, gesture_data: str):
        """Process gesture input and update dialogue state"""
        # Parse gesture data and update dialogue context
        if 'pointing' in gesture_data:
            self.dialogue_manager.state.context['gesture'] = 'pointing'
        elif 'wave' in gesture_data:
            self.dialogue_manager.state.context['gesture'] = 'greeting'
        elif 'stop' in gesture_data:
            self.dialogue_manager.state.context['gesture'] = 'stop'

    def _update_user_tracking(self):
        """Update tracking of users in the environment"""
        # In practice, this would use face recognition to identify users
        # For now, we'll simulate user detection
        detected_users = ['user_1']  # Simulated detection

        for user in detected_users:
            if user not in self.user_interaction_count:
                self.user_interaction_count[user] = 0
            self.user_interaction_count[user] += 1

            if self.current_user is None:
                self.current_user = user
                self.get_logger().info(f"New user detected: {user}")

    def _update_joint_attention(self, gaze_point: Tuple[float, float, float]):
        """Update joint attention system with human gaze"""
        # Update the joint attention system
        if hasattr(self.dialogue_manager, 'joint_attention_system'):
            self.dialogue_manager.joint_attention_system.update_human_attention(
                gaze_point, attention_confidence=0.9
            )

    def generate_speech_response(self, text: str, user_id: str = None):
        """Generate and publish speech response"""
        if not self.hri_enabled:
            return

        # Personalize response if user is known
        if user_id:
            personalized_text = self.personalization_system.get_personalized_response(
                user_id, text
            )
        else:
            personalized_text = text

        # Publish speech
        speech_msg = String()
        speech_msg.data = personalized_text
        self.speech_pub.publish(speech_msg)

    def execute_gesture(self, gesture_type: str):
        """Execute a robot gesture"""
        if not self.hri_enabled:
            return

        gesture_msg = String()
        gesture_msg.data = gesture_type
        self.gesture_pub.publish(gesture_msg)

    def set_face_expression(self, expression: str):
        """Set robot face expression"""
        face_msg = String()
        face_msg.data = expression
        self.face_display_pub.publish(face_msg)

    def get_user_trust_level(self, user_id: str) -> float:
        """Get trust level for a specific user"""
        return self.trust_model.get_trust_level(user_id)

    def get_personalized_behavior(self, user_id: str) -> Dict:
        """Get personalized behavior for a specific user"""
        return self.personalization_system.get_personalized_behavior(user_id)
```

## HRI Quality Assessment

### Interaction Quality Metrics

```python
class HRIQualityAssessor:
    def __init__(self):
        self.interaction_logs = []
        self.engagement_metrics = []
        self.satisfaction_scores = []

    def assess_interaction_quality(self, interaction_data: Dict) -> Dict:
        """Assess quality of human-robot interaction"""
        metrics = {
            'engagement_level': self._calculate_engagement(interaction_data),
            'responsiveness': self._calculate_responsiveness(interaction_data),
            'naturalness': self._calculate_naturalness(interaction_data),
            'satisfaction': self._calculate_satisfaction(interaction_data),
            'trust_building': self._calculate_trust_building(interaction_data)
        }

        # Store for analysis
        self.interaction_logs.append({
            'timestamp': time.time(),
            'metrics': metrics,
            'raw_data': interaction_data
        })

        return metrics

    def _calculate_engagement(self, data: Dict) -> float:
        """Calculate engagement level"""
        # Engagement factors: eye contact, response rate, interaction duration
        eye_contact_time = data.get('eye_contact_duration', 0)
        total_interaction_time = data.get('total_duration', 1)
        response_rate = data.get('response_rate', 0)
        turn_exchanges = data.get('turn_exchanges', 0)

        engagement = (eye_contact_time / total_interaction_time * 0.4 +
                     response_rate * 0.3 +
                     min(turn_exchanges / 5, 1.0) * 0.3)

        return min(engagement, 1.0)

    def _calculate_responsiveness(self, data: Dict) -> float:
        """Calculate responsiveness"""
        avg_response_time = data.get('avg_response_time', float('inf'))

        if avg_response_time < 2.0:
            return 1.0
        elif avg_response_time < 5.0:
            return 0.7
        elif avg_response_time < 10.0:
            return 0.4
        else:
            return 0.1

    def _calculate_naturalness(self, data: Dict) -> float:
        """Calculate naturalness of interaction"""
        # Naturalness factors: turn taking, gesture usage, language flow
        turn_taking_smoothness = data.get('turn_taking_smoothness', 0.5)
        gesture_usage = data.get('gesture_usage', 0.5)
        language_flow = data.get('language_flow', 0.5)

        naturalness = (turn_taking_smoothness * 0.4 +
                      gesture_usage * 0.3 +
                      language_flow * 0.3)

        return naturalness

    def _calculate_satisfaction(self, data: Dict) -> float:
        """Calculate user satisfaction"""
        # Satisfaction from explicit feedback or behavioral cues
        explicit_feedback = data.get('explicit_feedback', 0.5)
        behavioral_indicators = data.get('behavioral_indicators', 0.5)

        return (explicit_feedback * 0.7 + behavioral_indicators * 0.3)

    def _calculate_trust_building(self, data: Dict) -> float:
        """Calculate trust building effectiveness"""
        success_rate = data.get('task_success_rate', 0.5)
        safety_perception = data.get('safety_perception', 0.5)
        reliability_rating = data.get('reliability_rating', 0.5)

        trust_building = (success_rate * 0.5 +
                         safety_perception * 0.3 +
                         reliability_rating * 0.2)

        return trust_building

    def detect_interaction_problems(self, current_metrics: Dict) -> List[str]:
        """Detect problems in human-robot interaction"""
        problems = []

        if current_metrics['engagement_level'] < 0.3:
            problems.append('low_engagement')
        if current_metrics['responsiveness'] < 0.4:
            problems.append('poor_responsiveness')
        if current_metrics['naturalness'] < 0.4:
            problems.append('unnatural_interaction')
        if current_metrics['satisfaction'] < 0.5:
            problems.append('low_satisfaction')

        return problems

    def get_interaction_insights(self) -> Dict:
        """Get insights about HRI quality over time"""
        if not self.interaction_logs:
            return {'message': 'No interaction data available'}

        recent_logs = self.interaction_logs[-min(10, len(self.interaction_logs)):]

        insights = {
            'average_engagement': np.mean([log['metrics']['engagement_level'] for log in recent_logs]),
            'average_responsiveness': np.mean([log['metrics']['responsiveness'] for log in recent_logs]),
            'trend_improvement': len([log for log in recent_logs if log['metrics']['engagement_level'] > 0.7]) > len(recent_logs) * 0.5,
            'common_problems': self._get_common_problems(recent_logs),
            'satisfaction_trend': 'improving' if len(recent_logs) > 1 and
                                 recent_logs[-1]['metrics']['satisfaction'] > recent_logs[0]['metrics']['satisfaction'] else 'declining'
        }

        return insights

    def _get_common_problems(self, logs: List) -> List[str]:
        """Get most common problems in recent interactions"""
        all_problems = []
        for log in logs:
            metrics = log['metrics']
            if metrics['engagement_level'] < 0.3:
                all_problems.append('engagement')
            if metrics['responsiveness'] < 0.4:
                all_problems.append('responsiveness')
            if metrics['naturalness'] < 0.4:
                all_problems.append('naturalness')

        from collections import Counter
        problem_counts = Counter(all_problems)
        return [problem for problem, count in problem_counts.most_common(3)]
```

## Key Takeaways

- Human-robot interaction requires multiple communication modalities (speech, gestures, etc.)
- Natural language processing enables verbal communication and understanding
- Gesture recognition allows for non-verbal communication
- Social intelligence helps robots understand and respond to human social cues
- Trust modeling builds and maintains human-robot relationships
- Personalization adapts robot behavior to individual users
- Quality assessment ensures effective and satisfying interactions

## Next Steps

In the next chapter, we'll explore cognitive architectures and system integration, learning how to combine all the components we've studied into a cohesive AI robot brain system.