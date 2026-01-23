# Engineering Journal Entry

## Problem Definition and Scope Exploration

### Problem Statement

The core problem this project aims to solve is the reconstruction and evaluation of human-drawn trajectories performed in free space (“air writing”). Unlike traditional handwriting, air writing lacks a physical surface, resulting in increased noise, ambiguity in depth, and variability in stroke formation. The challenge is to accurately capture these trajectories, normalize them across users and drawing styles, and evaluate them against a known ground-truth representation.

Initially inspired by the Disney Channel wand intro, the problem generalizes well beyond entertainment. The same underlying challenge appears in gesture recognition, handwriting education, accessibility tools, and interactive human–computer interfaces. In particular, extending this system to language learning introduces stricter correctness constraints, such as stroke order, relative proportions, and directional accuracy, which elevates the technical difficulty and usefulness of the system.

Thus, the problem can be reframed as follows:

> How can a system robustly capture free-space drawing motion using vision-based methods and evaluate the accuracy of the resulting trajectory against symbolic templates (e.g., characters, shapes, or gestures) in a way that is both precise and user-tolerant?

Key subproblems include:

- Reliable trajectory acquisition under noisy conditions
    
- Segmentation of continuous motion into meaningful strokes
    
- Normalization of scale, rotation, and speed
    
- Definition of similarity metrics that align with human perception of correctness
    
- Providing interpretable feedback to users for learning and improvement
    

---

# Engineering Journal Entry

## Exploration of Possible Technical Approaches

### Overview

Several approaches were explored for capturing and evaluating free-space writing and drawing motions. These approaches vary in complexity, robustness, and suitability for handwriting analysis. The goal of this exploration was to identify methods that balance novelty, feasibility, and educational value while avoiding unnecessary hardware complexity.

---

### Approach 1: Pure Computer Vision–Based Tracking

This approach relies entirely on vision-based methods to detect and track a drawing instrument or body part, such as a finger, pen, or wand, using a single camera. The system extracts the position of the drawing point frame-by-frame and reconstructs a trajectory over time.

This method has the advantage of minimal setup and high accessibility, as it requires no additional hardware beyond a standard camera. However, it is sensitive to occlusion, lighting conditions, and depth ambiguity. Because air writing does not provide physical constraints, the resulting trajectories can be noisy and inconsistent, requiring substantial smoothing and filtering.

While feasible for shape drawing and gesture recognition, this approach becomes more challenging when applied to fine-grained handwriting accuracy, where small deviations can significantly affect perceived correctness.

---

### Approach 2: Explicit Tip Tracking via Physical Markers

In this approach, a physical drawing instrument (e.g., a wand or pen) is augmented with a clearly identifiable tip, such as a colored marker, LED, or reflective element. The system tracks this tip directly using computer vision techniques, producing a cleaner and more stable trajectory.

This method improves positional accuracy and simplifies trajectory reconstruction by reducing ambiguity in identifying the drawing point. It also shifts complexity away from advanced machine learning toward geometric and signal-processing techniques, which are more predictable and easier to validate.

The main drawback is the requirement for a physical prop, which slightly increases setup complexity. However, this tradeoff is acceptable given the improvement in data quality, particularly for handwriting evaluation where precision is critical.

---

### Approach 3: Hybrid Tracking (Context + Precision)

A hybrid approach combines coarse contextual tracking (e.g., hand or wand pose) with precise tracking of a designated tip. The contextual data provides robustness against occlusion and motion discontinuities, while the explicit tip tracking provides high-resolution trajectory data.

This fusion allows the system to better infer stroke boundaries, writing intent, and directional motion. Although this approach introduces additional engineering complexity, it offers the most flexibility and robustness, making it well-suited for both gesture-based interactions (such as spell casting) and detailed handwriting analysis.

This approach aligns strongly with the project’s goals of extensibility and educational feedback.

---

### Approach 4: Gesture Template Matching Inspired by Theme Park Systems

Commercial systems such as those used in theme parks rely on predefined gesture templates and tolerance envelopes rather than precise shape matching. These systems prioritize recognition of intent over exact geometric accuracy.

While this approach is effective for large-scale gestures and interactive experiences, it is less suitable for handwriting learning, where correctness depends on fine structural details. However, elements of this approach—such as directional tolerance and sequence-based matching—can be adapted to complement more precise trajectory comparison methods.

---

### Comparative Evaluation and Direction

Among the explored approaches, hybrid tracking with an explicit tip marker offers the best balance between robustness, accuracy, and feasibility. It enables clean trajectory capture without requiring specialized hardware and supports advanced evaluation metrics such as stroke-level comparison and temporal alignment.

This approach also scales naturally from entertainment-focused use cases (e.g., Disney-style drawing) to educational applications, such as evaluating the accuracy of handwritten characters in different languages.

---

### Conclusion

The exploration highlights that the central challenge is not merely tracking motion, but defining meaningful representations and comparison metrics for free-space writing. By focusing on trajectory normalization, stroke analysis, and perceptually aligned similarity measures, the system can remain technically manageable while achieving a high degree of novelty and applicability.

---

## Papers for Each Option (Literature Survey)
1. Vision-Based Air Writing / Free-Space Handwriting

Vision-Based Air-Writing Recognition System — Zhang et al., IEEE, 2013

Air-Writing Recognition Using Deep Convolutional Neural Networks — Li et al., 2017

Real-Time Vision-Based Finger Tracking for Air Writing — Xu et al., Pattern Recognition Letters

2. Explicit Marker / Tip Tracking (LED, Colored Tip, Pen)

Learning OpenCV — Bradski & Kaehler (color and marker tracking techniques)

Vision-Based Pen Tracking for Handwriting Recognition — Yeo et al., IEEE

Robust Marker-Based Motion Tracking Using a Single Camera — Sensors, 2019

3. Hybrid Tracking / Sensor Fusion

A Survey of Computer Vision-Based Human Motion Capture — Moeslund et al., CVIU

Hybrid Gesture Recognition Using Marker and Skeleton Tracking — Zhou et al.

4. Gesture Recognition / Theme-Park-Style Systems

Gestures Without Libraries, Toolkits, or Training — Wobbrock et al., CHI 2007

Improving Gesture Recognition Accuracy on Touch Screens — Vatavu et al.

5. Trajectory Comparison & Accuracy Metrics

Dynamic Time Warping Algorithm Optimization — Sakoe & Chiba

Hausdorff Distance for Shape Matching

Spline-Based Curve Matching

6. Handwriting Learning & Stroke Analysis (Language Extension)

Online and Offline Handwriting Recognition: A Comprehensive Survey — Plamondon & Srihari

Stroke-Based Character Evaluation for Handwriting Education

State of the Art in Online Handwriting Recognition — Tappert et al.

# Engineering Journal Entry - January 23, 2026
## Exploration of an Infrared (IR) + Computer Vision Hybrid Approach

### Motivation for IR + CV Integration

One of the fundamental challenges identified in free-space writing is the reliable localization of the drawing point under varying environmental conditions. Pure RGB computer vision approaches are sensitive to lighting changes, background clutter, occlusion, and color ambiguity. These limitations become increasingly problematic when the system is expected to evaluate fine-grained handwriting accuracy rather than coarse gesture recognition.

To address these issues, an alternative approach involving the integration of infrared (IR) sensing with conventional computer vision was explored. The motivation behind this approach is to decouple the task of identifying the drawing point from the complexities of the visible spectrum, while still leveraging camera-based spatial reasoning and visualization.

---

### Conceptual Overview of the IR + CV Approach

In an IR + CV system, the drawing instrument (e.g., wand or pen) is augmented with an IR-emitting or IR-reflective tip. An IR-sensitive camera or IR-filtered camera stream is used to isolate this signal, while a standard RGB camera (or RGB channel from the same camera) provides contextual information such as hand pose, scene geometry, and user feedback overlays.

The system thus operates on two complementary data streams:
- **IR channel**: Provides high-contrast, low-noise localization of the drawing tip  
- **RGB channel**: Provides contextual visual information for visualization, UI, and optional hand or body tracking  

The fusion of these channels allows the system to separate *precision tracking* from *semantic understanding*.

---

### Advantages of IR-Based Tip Tracking

Using IR significantly improves the robustness of drawing-point detection. Because IR light is largely invisible to the human eye and less common in ambient environments, the IR signal can be isolated with high confidence using simple thresholding techniques.

Key advantages include:
- Reduced sensitivity to ambient lighting variations  
- Minimal interference from background textures and colors  
- Clear separation between the drawing tip and the rest of the scene  
- Stable detection even during fast motion  

This makes IR particularly well-suited for handwriting evaluation, where small spatial deviations may meaningfully affect perceived correctness.

---

### Role of Computer Vision in the Hybrid System

While IR provides accurate positional data, computer vision remains essential for higher-level reasoning and interaction. CV techniques can be used to:
- Infer stroke boundaries using motion cues  
- Normalize trajectories based on hand orientation or writing direction  
- Provide real-time visual feedback and overlays  
- Support optional gesture recognition modes  

Importantly, the CV component does not need to solve the hardest tracking problem, allowing simpler and more reliable algorithms to be used.

---

### Data Fusion and System Design Considerations

The IR + CV approach introduces the need for data fusion between two sensing modalities. However, this fusion can be lightweight rather than complex. In many cases, the IR-derived coordinates can be treated as the authoritative drawing trajectory, while CV-derived data is used only to augment interpretation.

Potential fusion strategies include:
- Temporal alignment of IR points with RGB frames  
- Using CV-derived motion cues to segment strokes in IR trajectories  
- Applying filtering (e.g., exponential smoothing or Kalman filtering) to the IR signal before comparison  

This design avoids full sensor fusion complexity while still benefiting from multi-modal sensing.

---

### Comparison with Other Approaches

Compared to pure CV tracking, the IR + CV approach offers substantially improved robustness and precision, especially in cluttered or poorly lit environments. Compared to marker-based RGB tracking, IR provides better signal isolation without relying on color contrast, which may vary across environments.

Unlike theme-park-style gesture systems, which emphasize intent recognition over exact geometric accuracy, IR + CV supports detailed shape analysis and stroke-level evaluation. This makes it more appropriate for handwriting learning applications.

---

### Applicability to Language Learning and Educational Use

The IR + CV approach aligns strongly with the language-learning extension of the project. Accurate trajectory capture enables:
- Stroke order validation  
- Proportion and curvature analysis  
- Directional correctness evaluation  
- Visual feedback highlighting specific deviations  

Because the IR signal provides consistent data across users, comparisons against symbolic templates become more reliable and interpretable.

---

### Feasibility and Practical Constraints

From an implementation perspective, the IR + CV approach remains feasible within the scope of a senior design project. It does not require specialized motion-capture hardware and can be implemented using consumer-grade cameras and simple IR emitters or reflective materials.

The added hardware complexity is minimal, while the gains in data quality significantly reduce downstream algorithmic difficulty.

---

### Conclusion

The integration of infrared sensing with computer vision represents a principled compromise between robustness and complexity. By using IR for precise trajectory acquisition and CV for contextual understanding and feedback, the system achieves a modular and extensible design. This approach preserves novelty while remaining technically manageable and well-aligned with both entertainment-driven and educational use cases.
