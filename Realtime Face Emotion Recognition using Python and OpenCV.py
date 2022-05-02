#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[6]:


from deepface import DeepFace 


# In[7]:


img = cv2.imread('happy-boy-smile.jpg')


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


plt.imshow(img) ##BGR


# In[5]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) ##BGR to RGB


# In[10]:


predictions = DeepFace.analyze(img)


# In[11]:


predictions


# In[12]:


type(predictions)


# In[13]:


predictions['dominant_emotion']


# we are trying to draw a rectangle across the face

# In[14]:


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[15]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #print(face.Cascade.empty())
faces = faceCascade.detectMultiScale(gray,1.1,4)

  #Draw a rectangle around the face
for(x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# In[16]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[20]:


font = cv2.FONT_HERSHEY_SIMPLEX

  # Use putText() method for
    # inserting text on video
cv2.putText(img,
           predictions['dominant_emotion'],
           (50,50),
           font, 3,
           (0, 0, 255),
           2,
           cv2.LINE_4) ;


# In[21]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[22]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[23]:


img = cv2.imread('1.jpg')


# In[24]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[25]:


predictions = DeepFace.analyze(img)


# In[26]:


predictions


# In[27]:


img = cv2.imread('images.jpg')


# In[28]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[29]:


predictions = DeepFace.analyze(img)


# In[30]:


predictions


# In[31]:


font = cv2.FONT_HERSHEY_SIMPLEX

  # Use putText() method for
    # inserting text on video
cv2.putText(img,
           predictions['dominant_emotion'],
           (50,50),
           font, 3,
           (0, 0, 255),
           2,
           cv2.LINE_4) ;


# In[32]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[33]:


img = cv2.imread('images.jpg')


# In[34]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[35]:


predictions = DeepFace.analyze(img)


# In[36]:


font = cv2.FONT_HERSHEY_SIMPLEX

  # Use putText() method for
    # inserting text on video
cv2.putText(img,
           predictions['dominant_emotion'],
           (0,50),
           font, 1,
           (0, 0, 255),
           2,
           cv2.LINE_4) ;


# In[37]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# Real time video demo for Face Emotion Recognition

# In[1]:


import cv2

from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
# check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    
while True:
    ret,frame = cap.read() ## read one image from a video
    
    result = DeepFace.analyze(frame, actions = ['emotion'])
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    
    # Draw a rectangle around the faces
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
        
    #USE PUTtEXT() METHOD FOR
    #inserting text on video
    cv2.putText(frame,
                result['dominant_emotion'],
                (50,50),
                font, 3,
                (0, 0, 255),
                2,
                cv2.LINE_4)
    cv2.imshow('Demo video',frame)
        
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




