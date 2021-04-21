{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Install and Import Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install mediapipe opencv-python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import mediapipe as mp\n",
    "import cv2\n",
    "import numpy as np\n",
    "import uuid\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mp_drawing = mp.solutions.drawing_utils\n",
    "mp_hands = mp.solutions.hands"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Draw Hands\n",
    "<img src=https://i.imgur.com/qpRACer.png />"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: \n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        \n",
    "        # BGR 2 RGB\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        \n",
    "        # Flip on horizontal\n",
    "        image = cv2.flip(image, 1)\n",
    "        \n",
    "        # Set flag\n",
    "        image.flags.writeable = False\n",
    "        \n",
    "        # Detections\n",
    "        results = hands.process(image)\n",
    "        \n",
    "        # Set flag to true\n",
    "        image.flags.writeable = True\n",
    "        \n",
    "        # RGB 2 BGR\n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "        \n",
    "        # Detections\n",
    "        print(results)\n",
    "        \n",
    "        # Rendering results\n",
    "        if results.multi_hand_landmarks:\n",
    "            for num, hand in enumerate(results.multi_hand_landmarks):\n",
    "                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, \n",
    "                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),\n",
    "                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),\n",
    "                                         )\n",
    "            \n",
    "        \n",
    "        cv2.imshow('Hand Tracking', image)\n",
    "\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mp_drawing.DrawingSpec??"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Output Images "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "os.mkdir('Output Images')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: \n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        \n",
    "        # BGR 2 RGB\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        \n",
    "        # Flip on horizontal\n",
    "        image = cv2.flip(image, 1)\n",
    "        \n",
    "        # Set flag\n",
    "        image.flags.writeable = False\n",
    "        \n",
    "        # Detections\n",
    "        results = hands.process(image)\n",
    "        \n",
    "        # Set flag to true\n",
    "        image.flags.writeable = True\n",
    "        \n",
    "        # RGB 2 BGR\n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "        \n",
    "        # Detections\n",
    "        print(results)\n",
    "        \n",
    "        # Rendering results\n",
    "        if results.multi_hand_landmarks:\n",
    "            for num, hand in enumerate(results.multi_hand_landmarks):\n",
    "                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, \n",
    "                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),\n",
    "                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),\n",
    "                                         )\n",
    "            \n",
    "        # Save our image    \n",
    "        cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)\n",
    "        cv2.imshow('Hand Tracking', image)\n",
    "\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "hands",
   "language": "python",
   "name": "hands"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
