# 🎬 Krishi-AI: Complete Video Presentation & Demo Script
**Role:** Solo Presenter / Developer  
**Estimated Runtime:** 5 to 7 Minutes  
**Target Setup:** Screen recording showing browser at `http://127.0.0.1:5000` + optional picture-in-picture webcam  

---

## 📋 Pre-Recording Checklist
1. **Server Running**: Ensure `python app.py` is active and reachable at `http://127.0.0.1:5000`.
2. **Audio Check**: Good microphone, quiet room, audio output unmuted so the browser's "Read Aloud" voice is clearly captured.
3. **Display Settings**: Browser zoom set to 100% or 110% at 1080p resolution for clean readability.
4. **Tabs Ready**: Have the browser open on the Krishi-AI homepage (`http://127.0.0.1:5000`).

---

## ⏱️ Timeline & Scene Outline

```
  [0:00 - 0:50] 🌟 ACT 1: Hook, Motivation & The "Why"
  [0:50 - 2:00] 🧠 ACT 2: System Architecture & The 4 AI Pillars
  [2:00 - 4:45] 🖥️ ACT 3: Hands-On Live Website Walkthrough
      ├── 2:00 - 2:45: Case 1: Wheat Leaf Rust & 2-Stage Autonomous Decision
      ├── 2:45 - 3:30: OpenCV 4-Panel Lesion Visualizer & Clinical Tabs
      ├── 3:30 - 4:00: Case 2: Rice Blast & Tri-Lingual Support (Hindi / Bengali)
      └── 4:00 - 4:45: Web Speech Voice Narration & Live Farmer Q&A Chat
  [4:45 - 5:45] 🍓 ACT 4: Edge Deployment (Raspberry Pi 4) & Offline Architecture
  [5:45 - 6:15] 🎯 ACT 5: Conclusion & Future Outlook
```

---

## 🎬 Detailed Script: Scene by Scene

---

### 🌟 ACT 1: Hook, Motivation & The Real-World Problem (0:00 – 0:50)

**[ON SCREEN]**  
*Start with camera on yourself, or showing the hero section of the Krishi-AI web interface (`http://127.0.0.1:5000`). The dark glassmorphism UI with ambient green glow is visible.*

**[SPOKEN SCRIPT]**  
> *"Hello everyone! Today, global agriculture is facing a critical challenge: staple cereal crops like Rice and Wheat account for over 50% of the world's daily caloric intake. Yet every single year, smallholder farmers lose between 20% to 40% of their total harvest to fungal diseases, bacterial blights, and aggressive insect pests.*
>
> *Now, you might ask: aren't there smartphone apps for crop disease detection? Yes, there are. But almost all of them suffer from two fatal flaws:*
> 1. *First, they depend heavily on cloud APIs, meaning they fail the moment a farmer steps into a remote field with zero internet.*
> 2. *Second, they are trained on artificial laboratory datasets—photographs of single cut leaves placed on plain white paper sheets. In an authentic field with messy soil, varying sunlight, and tangled leaves, those models collapse.*
>
> *Today, I am proud to present **Krishi-AI**: an autonomous, clinical crop pathology and advisory system designed to run **100% offline on edge devices** like the Raspberry Pi 4 and mobile phones."*

---

### 🧠 ACT 2: System Architecture & The 4 Technological Pillars (0:50 – 2:00)

**[ON SCREEN]**  
*Scroll gently down to show the Hero Banner tech pills: `EfficientNet-B0`, `Hierarchical Inference`, `OpenCV Severity`, `SQLite Treatments`.*

**[SPOKEN SCRIPT]**  
> *"Krishi-AI is not just a classification model; it is an end-to-end clinical diagnostic pipeline powered by four interconnected pillars:*
>
> * **Pillar 1: Purified In-Field Vision**: *We trained a high-efficiency **EfficientNet-B0** classifier on 28,833 authentic in-field images across 29 classes—13 Wheat conditions, 3 Rice conditions, and 13 Agricultural Pest classes. We implemented an automated background purification filter that purged over 1,900 white-card studio images, achieving **89.08% validation accuracy** and an **88.61% macro F1-score** exclusively on real-world field images.*
>
> * **Pillar 2: Autonomous Two-Stage Hierarchical Decision**: *Unlike naive classifiers that ask the farmer to pick the crop from a dropdown, Krishi-AI decides autonomously. Stage 1 determines whether the specimen is **Wheat**, **Rice**, or an **Agricultural Pest**. Once the host is established, Stage 2 isolates the exact pathology.*
>
> * **Pillar 3: Physical OpenCV Folier Lesion Quantification**: *Detecting a disease name isn't enough for treatment. A farmer needs to know how bad it is. We built a multi-color space OpenCV segmentation engine using HSV, CIE-Lab, and Excess Green index (ExG) that measures the exact percentage of infected foliar surface area and maps it to international **IRRI SES** and **Modified Cobb** severity grades.*
>
> * **Pillar 4: Offline Clinical SQLite Knowledge Base & Llama 3.2**: *Connected to a verified, zero-cloud database containing ICAR, IRRI, and USDA-standardized remedies: calibrated chemical dosages, pre-harvest safety intervals, organic bio-pesticides, and resistant cultivars—synthesized by our agronomic reasoner, **Dr. Krishi**.*
>
> *Let's see it in action!"*

---

### 🖥️ ACT 3: Hands-On Live Website Walkthrough (2:00 – 4:45)

#### Scene 3A: One-Click Field Demo & Two-Stage Autonomous Decision (2:00 – 2:45)

**[ON SCREEN ACTION]**  
*Hover over the left panel. Point the cursor to the dropzone. Then move down to the **One-Click Authentic Field Samples** and click on **Wheat Leaf Rust**.*

**[SPOKEN SCRIPT]**  
> *"Here is the Krishi-AI interactive dashboard. A farmer can take a photo with their phone camera, drag and drop an image, or paste from clipboard. Notice our 'Auto-decide' feature is active.*
>
> *Let's test our first field sample: **Wheat Leaf Rust**.*
>
> *(Click 'Wheat Leaf Rust')*
>
> *Watch the progressive scanner execute in real time. And look at this result!*
>
> *Notice our **Two-Stage Decision Deck**:*
> * *In **Stage 1 (Crop Decision)**, the system did not guess—it evaluated the probability across all species. It judged the plant as **Wheat (Triticum aestivum)** with **100% certainty**, with 0% attributed to Rice or Pests.*
> * *Having locked in Wheat, **Stage 2 (Pathology Detection)** immediately isolated the condition as **Stem/Leaf Rust** with **99.2% certainty**, flagging an active pathological infection.*
> * *Notice it also displays the runner-up candidate, showing full clinical transparency."*

---

#### Scene 3B: OpenCV 4-Panel Canvas & Prescription Treatments (2:45 – 3:30)

**[ON SCREEN ACTION]**  
*Scroll slightly down to the **Physical Diagnostic Visualization** card. Click on the **4-Panel Canvas** tab, then toggle back to **Annotated Overlay**.*

**[SPOKEN SCRIPT]**  
> *"Now look below at **Stage 3: Foliar Lesion Spread**. OpenCV calculated that exactly **18.37%** of the leaf surface is covered in necrotic rust pustules, placing this crop at **Grade 2 (Moderate Severity)**.*
>
> *Let's inspect the physical evidence. Here in the Visual Inspection card:*
> * *The **Annotated Overlay** draws precise bounding contours around every active rust pustule.*
> * *Clicking **4-Panel Canvas** shows our full computer vision pipeline: the original foliage, the multi-space green leaf mask, the isolated lesion mask, and the combined HUD overlay.*
>
> *Now, what should the farmer actually do? We don't give vague advice. Let's look at the **Clinical Remedies Tabs**:*
> * *Under **Curative Chemical**, we have **Propiconazole 25% EC (Tilt / Bumper)** with exact calibrated dosage: **1 ml per liter of water (200 ml per acre)**, application timing, and critical safety rules like a 30-day pre-harvest interval.*
> * *If the grower is organic, the **Organic / Bio-Control** tab prescribes bio-agents like *Trichoderma viride* or Neem extract.*
> * *And the **Cultural & Seeds** tab lists certified rust-resistant cultivars like **HD-3086** and **PBW-550** for subsequent crop cycles."*

---

#### Scene 3C: Rice Blast & Tri-Lingual Localization (3:30 – 4:00)

**[ON SCREEN ACTION]**  
*Scroll up. Click on the demo sample **Rice Leaf Blast**. Wait 1 second for the result to update. Then, in the top navigation bar, click **हिन्दी (Hindi)**, and then **বাংলা (Bengali)**.*

**[SPOKEN SCRIPT]**  
> *"Now let's switch crops completely without changing any settings. I'll click **Rice Leaf Blast**.*
>
> *(Click 'Rice Leaf Blast')*
>
> *Immediately, Stage 1 shifts its autonomous judgment: **Rice (Oryza sativa)** at **99.5% certainty**! And Stage 2 identifies **Leaf Blast (Magnaporthe oryzae)**.*
>
> *Notice the OpenCV lesion spread: **42.08%**! That is severe necrotic collapse—Grade 3—triggering an urgent 24-to-48-hour action window with systemic tricyclazole recommendations.*
>
> *Now, real farmers in India, Bangladesh, and across South Asia speak regional languages. Watch this: in the top right, I click **हिन्दी (Hindi)**.*
>
> *(Click 'हिन्दी')*
>
> *Every single element—headers, diagnostic stages, chemical instructions, and urgency warnings—instantly translates into clear Hindi:*
> *'पहला चरण: फसल का निर्णय - धान'*
> *'दूसरा चरण: रोग की पहचान - लीफ ब्लास्ट'*
>
> *And with one click on **বাংলা (Bengali)**, it translates into fluent Bengali for farmers across West Bengal and Bangladesh."*

---

#### Scene 3D: Audio Voice Narration & Live Farmer Q&A Chat (4:00 – 4:45)

**[ON SCREEN ACTION]**  
*Switch back to **EN** (or keep Hindi/Bengali). Click on the **Dr. Krishi Advisory** tab. Click the **Read Aloud** button (let it speak for 3-5 seconds, then stop it). Then type a question into the chat input: `"Can I spray if it rains tonight?"` and click **Ask**.*

**[SPOKEN SCRIPT]**  
> *"Many rural farmers have low literacy levels. To ensure total accessibility, we integrated in-browser **Web Speech API text-to-speech narration**. Let's click **Read Aloud**:*
>
> *(Click 'Read Aloud' — let the browser voice speak: "The crop foliage has been evaluated as Rice Leaf Blast with diagnostic confidence...")*
>
> *(Click voice button to pause)*
>
> *Now, farmers always have real-world follow-up doubts. Let's ask Dr. Krishi:*
>
> *(Type in chat: 'Can I spray if it rains tonight?' and submit)*
>
> *Look at Dr. Krishi's instant response:*
> *'🌧️ Spray & Weather Protocol: Do NOT apply foliar sprays if rain is expected within 4–6 hours. Rainfall washes off fungicides before absorption, causing chemical wastage and environmental runoff. Best spraying window: calm early mornings (6:30–9:30 AM).'*
>
> *It's completely grounded in verified agronomy, with zero hallucination."*

---

### 🍓 ACT 4: Edge Deployment on Raspberry Pi 4 (4:45 – 5:45)

**[ON SCREEN]**  
*Switch to terminal or show a slide / diagram of Raspberry Pi 4 architecture (or scroll down to the Edge Deployment section of the README).*

**[SPOKEN SCRIPT]**  
> *"Now, the most critical question: how does this scale in the real world?*
>
> *Most modern multi-modal AI systems require high-end server clusters with multi-gigabyte cloud APIs. Krishi-AI is engineered from the ground up for **Zero-Cloud Edge Computing**:*
>
> 1. *Our EfficientNet-B0 backbone has been optimized for ARM NEON vector instructions and INT8 quantization, shrinking the memory footprint to just **~4.2 megabytes**.*
> 2. *OpenCV image processing and SQLite query transactions take less than **25 milliseconds**.*
> 3. *The entire application runs comfortably within **60 megabytes of RAM**, making it 100% operational on a standard **$35 Raspberry Pi 4** or directly inside a mobile smartphone app without any internet connection.*
>
> *A field extension worker or village cooperative can mount this system onto a handheld device, a field kiosk, or even an agricultural drone for localized edge screening."*

---

### 🎯 ACT 5: Conclusion & Call to Action (5:45 – 6:15)

**[ON SCREEN]**  
*Back to the live website dashboard at `http://127.0.0.1:5000` showing the full results card.*

**[SPOKEN SCRIPT]**  
> *"In summary, Krishi-AI bridges the gap between state-of-the-art computer vision and ground-level agricultural reality:*
> * It eliminates laboratory bias by training exclusively on purified in-field data.
> * It eliminates guesswork with autonomous two-stage crop-and-disease decision making.
> * It measures real physical severity percentages with OpenCV.
> * And it delivers verified, actionable agronomic prescriptions in local languages with zero cloud dependency.
>
> *Thank you very much! The complete code, trained models, database, and web application are open-sourced on GitHub. Feel free to try it out!"*

---

## 💡 Quick Tips for the Presenter

| Timing | Action | Tip |
| :--- | :--- | :--- |
| **0:15** | Face camera / show homepage | Speak with energy; state the problem clearly. |
| **2:10** | Clicking demo sample | Don't rush; let the audience see the 2-stage cards animate. |
| **3:10** | Toggling 4-Panel Canvas | Highlight that this is actual computer vision, not just a black-box label. |
| **3:45** | Changing Language | Language switching creates a memorable "wow" moment for judges. |
| **4:20** | Chat Q&A | Show that Dr. Krishi understands practical farmer problems like rain and timing. |
