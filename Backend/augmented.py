import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import hashlib
import math # Import for more precise distance calculations

app = Flask(__name__)
CORS(app)

class FaceShapeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # Increased detection confidence slightly for cleaner results
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6
        )
        
        self.face_shapes = {
            'oval': 'Oval',
            'round': 'Round', 
            'square': 'Square',
            'heart': 'Heart',
            'diamond': 'Diamond',
            'oblong': 'Oblong',
            'triangle': 'Triangle'
        }
        
        self.analysis_cache = {}
        
        # Landmark indices (Kept the same as requested)
        self.landmark_indices = {
            'forehead_left': 54,
            'forehead_right': 284,
            'forehead_center': 10,
            'chin': 152,
            'jaw_left': 172,
            'jaw_right': 397,
            'cheek_left': 234,
            'cheek_right': 454,
            'nose_tip': 1,
            'left_eye_left': 33,
            'right_eye_right': 263,
            'jaw_mid_left': 136,
            'jaw_mid_right': 365,
            'left_eyebrow_upper': 65,
            'right_eyebrow_upper': 295
        }

    def get_image_hash(self, image):
        return hashlib.md5(image.tobytes()).hexdigest()

    def get_face_landmarks(self, image):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
                
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            landmark_points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append((x, y))
            return landmark_points
            
        except Exception as e:
            print(f"Error in landmark detection: {e}")
            return None

    def calculate_face_ratios(self, landmarks):
        if not landmarks:
            return None
        
        try:
            idx = self.landmark_indices
            
            # --- Measurement Calculations ---
            
            # Forehead Width (Distance between points 54 and 284)
            forehead_width = np.linalg.norm(
                np.array(landmarks[idx['forehead_left']]) - 
                np.array(landmarks[idx['forehead_right']])
            )
            
            # Cheek Width (Distance between points 234 and 454 - Widest part of the face)
            cheek_width = np.linalg.norm(
                np.array(landmarks[idx['cheek_left']]) - 
                np.array(landmarks[idx['cheek_right']])
            )
            
            # Jaw Width (Distance between points 172 and 397 - Jaw corners)
            jaw_width = np.linalg.norm(
                np.array(landmarks[idx['jaw_left']]) - 
                np.array(landmarks[idx['jaw_right']])
            )
            
            # Face Length (Distance between points 10 and 152 - Forehead center to Chin)
            face_length = np.linalg.norm(
                np.array(landmarks[idx['forehead_center']]) - 
                np.array(landmarks[idx['chin']])
            )
            
            # Jaw Mid Width (Used for jaw shape progression)
            jaw_mid_width = np.linalg.norm(
                np.array(landmarks[idx['jaw_mid_left']]) - 
                np.array(landmarks[idx['jaw_mid_right']])
            )
            
            # --- Ratio Calculations ---
            
            # 1. Length-to-Max-Width Ratio
            max_width = max(forehead_width, cheek_width, jaw_width)
            length_width_ratio = face_length / max_width if max_width > 0 else 1.0
            
            # 2. Jaw-to-Forehead Ratio
            jaw_forehead_ratio = jaw_width / forehead_width if forehead_width > 0 else 1.0
            
            # 3. Cheek-to-Jaw Ratio (Cheekbones relative to jaw)
            cheek_jaw_ratio = cheek_width / jaw_width if jaw_width > 0 else 1.0
            
            # 4. Forehead-to-Cheek Ratio (For Heart/Diamond)
            forehead_cheek_ratio = forehead_width / cheek_width if cheek_width > 0 else 1.0
            
            # 5. Jaw Progression (Used in original code, kept for consistency)
            jaw_progression = jaw_width / jaw_mid_width if jaw_mid_width > 0 else 1.0
            
            return {
                'length_width_ratio': length_width_ratio,
                'jaw_forehead_ratio': jaw_forehead_ratio,
                'cheek_jaw_ratio': cheek_jaw_ratio,
                'forehead_cheek_ratio': forehead_cheek_ratio, # New ratio
                'jaw_progression': jaw_progression,
                'face_length': face_length,
                'jaw_width': jaw_width,
                'forehead_width': forehead_width,
                'cheek_width': cheek_width,
                'jaw_mid_width': jaw_mid_width
            }
            
        except Exception as e:
            print(f"Error calculating ratios: {e}")
            return None

    def classify_face_shape(self, ratios):
        """Refined classification logic based on established metrics."""
        if not ratios:
            return "Unknown"
            
        try:
            lw_ratio = ratios['length_width_ratio']
            jf_ratio = ratios['jaw_forehead_ratio']
            cj_ratio = ratios['cheek_jaw_ratio']
            fc_ratio = ratios['forehead_cheek_ratio']
            
            # --- STEP 1: LONG VS. SHORT FACES ---
            
            if lw_ratio > 1.4:
                # Long faces (Oval, Oblong, possibly long Heart/Diamond)
                if lw_ratio > 1.6: # Very long
                    if abs(jf_ratio - 1.0) < 0.1:
                        shape = 'oblong' # Uniform width
                    else:
                        shape = 'diamond' # Likely wider cheekbones than jaw/forehead
                elif fc_ratio < 0.9:
                    shape = 'heart' # Wide forehead, narrow jaw
                elif 1.35 < lw_ratio <= 1.55 and 0.85 < jf_ratio < 1.05 and 0.95 < cj_ratio < 1.15:
                    shape = 'oval' # Balanced medium length
                else:
                    shape = 'oblong'
            
            elif lw_ratio < 1.2:
                # Short/Wide faces (Round, Square)
                if abs(jf_ratio - 1.0) < 0.1:
                    # Uniform width at jaw and forehead
                    shape = 'round' 
                elif jf_ratio > 1.1:
                    shape = 'triangle' # Wider jaw than forehead
                else:
                    shape = 'square' # Wide and structured jaw/forehead
            
            else:
                # Medium length faces (Square, Diamond, Heart, Triangle, Oval)
                
                # Check for Diamond (Cheekbones widest)
                if cj_ratio > 1.1:
                    if fc_ratio < 0.9:
                         shape = 'diamond'
                    else:
                        shape = 'heart' # Transition case
                
                # Check for Triangle (Jaw widest)
                elif jf_ratio > 1.1:
                    shape = 'triangle'
                
                # Check for Heart (Forehead widest, jaw narrow)
                elif fc_ratio > 1.05 and jf_ratio < 0.9:
                    shape = 'heart'
                    
                # Check for Square (Uniform/wide width and length)
                elif abs(jf_ratio - 1.0) < 0.15 and lw_ratio < 1.35:
                    shape = 'square'
                
                # Default to Oval (Balanced proportions)
                else:
                    shape = 'oval'

            # --- STEP 2: Final Review (Minimal) ---
            
            # Differentiate Round and Square when ratios are close to 1
            if shape == 'square' and ratios['jaw_progression'] > 1.25:
                # High jaw progression indicates a sharp corner, confirming Square
                pass 
            elif shape == 'square' and ratios['jaw_progression'] < 1.1:
                # Rounded jaw corners
                shape = 'round' 
                
            return self.face_shapes.get(shape, 'Unknown')
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return "Unknown"

    def calculate_confidence(self, ratios, detected_shape):
        """Calculate confidence based on how well ratios match ideal ranges (More sensitive)"""
        
        # NOTE: Defined ideal ranges are for the benefit of the confidence function.
        ideal_ranges = {
            'Oval':     {'lw_center': 1.45, 'lw_dev': 0.15, 'jf_center': 0.95, 'jf_dev': 0.1, 'cj_center': 1.0, 'cj_dev': 0.1},
            'Round':    {'lw_center': 1.1,  'lw_dev': 0.1,  'jf_center': 1.0,  'jf_dev': 0.1, 'cj_center': 1.0, 'cj_dev': 0.1}, 
            'Square':   {'lw_center': 1.25, 'lw_dev': 0.15, 'jf_center': 1.05, 'jf_dev': 0.1, 'cj_center': 1.05, 'cj_dev': 0.1},
            'Heart':    {'lw_center': 1.5,  'lw_dev': 0.2,  'jf_center': 0.8,  'jf_dev': 0.1, 'cj_center': 1.15, 'cj_dev': 0.15},
            'Diamond':  {'lw_center': 1.4,  'lw_dev': 0.15, 'jf_center': 0.9,  'jf_dev': 0.1, 'cj_center': 1.25, 'cj_dev': 0.15},
            'Oblong':   {'lw_center': 1.7,  'lw_dev': 0.2,  'jf_center': 1.0,  'jf_dev': 0.1, 'cj_center': 1.0, 'cj_dev': 0.1},
            'Triangle': {'lw_center': 1.3,  'lw_dev': 0.15, 'jf_center': 1.2,  'jf_dev': 0.15,'cj_center': 0.9, 'cj_dev': 0.1},
        }
        
        if detected_shape not in ideal_ranges:
            return 75
        
        ideal = ideal_ranges[detected_shape]
        lw_ratio = ratios['length_width_ratio']
        jf_ratio = ratios['jaw_forehead_ratio']
        cj_ratio = ratios['cheek_jaw_ratio']
        
        # Calculate how close ratios are to the center of the ideal range
        lw_score = 100 - (abs(lw_ratio - ideal['lw_center']) / ideal['lw_dev']) * 30
        jf_score = 100 - (abs(jf_ratio - ideal['jf_center']) / ideal['jf_dev']) * 30
        cj_score = 100 - (abs(cj_ratio - ideal['cj_center']) / ideal['cj_dev']) * 30
        
        confidence = (lw_score + jf_score + cj_score) / 3
        
        # Set a reasonable floor and ceiling for the score
        return min(99.9, max(50, confidence))

    # Existing calculate_score is not strictly needed anymore, 
    # but kept for potential debugging or future use
    def calculate_score(self, value, min_val, max_val):
        ideal_center = (min_val + max_val) / 2
        ideal_range = max_val - min_val
        distance = abs(value - ideal_center)
        
        if distance <= ideal_range / 2:
            score = 100 - (distance / (ideal_range / 2)) * 25
        else:
            score = max(0, 75 - (distance - ideal_range / 2) * 20)
            
        return score

    def analyze_image(self, image, gender="male"):
        image_hash = self.get_image_hash(image)
        
        if image_hash in self.analysis_cache:
            return self.analysis_cache[image_hash]
        
        landmarks = self.get_face_landmarks(image)
        
        if not landmarks:
            return None
        
        ratios = self.calculate_face_ratios(landmarks)
        if not ratios:
            return None
        
        face_shape = self.classify_face_shape(ratios)
        confidence = self.calculate_confidence(ratios, face_shape)
        
        result = {
            'face_shape': face_shape,
            'confidence': round(confidence, 1),
            'measurements': {
                'face_length': round(ratios['face_length'], 1),
                'face_width': round(max(ratios['forehead_width'], ratios['jaw_width']), 1),
                'length_width_ratio': round(ratios['length_width_ratio'], 2),
                'jaw_forehead_ratio': round(ratios['jaw_forehead_ratio'], 2),
                'cheek_jaw_ratio': round(ratios['cheek_jaw_ratio'], 2)
            },
            'debug_ratios': {
                'length_width': round(ratios['length_width_ratio'], 2),
                'jaw_forehead': round(ratios['jaw_forehead_ratio'], 2),
                'cheek_jaw': round(ratios['cheek_jaw_ratio'], 2)
            }
        }
        
        self.analysis_cache[image_hash] = result
        return result

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Face Shape API is running"})

@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400

        image_data = data['image']
        gender = data.get('gender', 'male')

        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image data'
            }), 400

        # Resize image for consistent processing (Kept original logic)
        height, width = image.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = 800
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        detector = FaceShapeDetector()
        result = detector.analyze_image(image, gender)

        if result:
            return jsonify({
                'success': True,
                'data': result
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No face detected in the image'
            })

    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Face Shape Analysis API...")
    print("📍 Backend URL: http://localhost:5000")
    print("⚡ Features: 7 Face Shapes, Advanced Measurements, Personalized Recommendations")
    app.run(debug=True, host='0.0.0.0', port=5000)