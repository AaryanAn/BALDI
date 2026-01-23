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
