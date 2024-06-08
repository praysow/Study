def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    
if x_center is not None and y_center is not None:
    speak(object_details)
    print(object_details)