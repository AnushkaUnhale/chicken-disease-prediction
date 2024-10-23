import streamlit as st
import tensorflow as tf
#import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="PoultryMitra | DiseasePrediction", layout='wide', page_icon="img/favicon.ico")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

#@st.cache(allow_output_mutation=True)
@st.cache_data
def load_model():
  model = tf.keras.models.load_model("C:/Users/anush/Desktop/BE Project/model/mobilenetV2/mobilenetv2.h5", compile = False) 
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

# # Apply the styling using st.markdown
# st.markdown(css, unsafe_allow_html=True)
st.markdown("<h1 style='margin:0; font-size: 80px; text-align: center; color: green;'>आपल्या कोंबडीचे आरोग्य तपासा</h1>", unsafe_allow_html=True)

st.title('')
file = st.file_uploader("तुमच्या कोंबडीच्या विष्ठेची चित्र येथे अपलोड करा", type=["jpg", "png", "jpeg"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_predict(upload_image, model):
        classes = {'Coccidiosis': 0, 'Healthy': 1, 'NCD': 2, 'Salmonella': 3}


        img = tf.keras.preprocessing.image.load_img(
          upload_image, 
          target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array*1/255.)
        score = tf.nn.softmax(predictions[0])
        pred_class=[j for j in classes if classes[j] == np.argmax(score)][0]
        return pred_class, round(100 * np.max(score),2)

if file is None:
    st.warning("Please upload an image of your chicken's feces")
else:
    audio_file = open("./audio/sample.mp3","rb")

    class_messages = {
          'Coccidiosis': "तुमची कोंबडी कदाचित रक्ती हगवण(Coccidiosis) ग्रस्त असेल.",
          'Healthy': "तुमच्या कोंबडीच्या विष्ठा निरोगी(Healthy) आहे!",
          'NCD': "तुमच्या कोंबडीला रानीखेत(Newcastle) रोग आहे असे दिसते.",
          'Salmonella': "साल्मोनेला (Salmonella) संसर्ग आढळला."
        }

    class_diagnosis = {
          'Coccidiosis': ["Coccidiosis is one of the most common diseases affecting chickens worldwide. It's caused by a single-celled parasite called Eimeria, which damages the chicken's intestines.  Coccidiosis can cause weight loss, poor growth, decreased egg production, and even death in severe cases."],
          'Healthy': [""],
          'NCD': ["Newcastle disease is a highly contagious viral infection that affects chickens and other birds. It can cause a range of symptoms, from mild to severe, depending on the strain of the virus and the bird's health. The virus spreads through direct contact with infected birds, their droppings, or contaminated equipment. "],
          'Salmonella': ["Salmonella is a bacterial infection that can affect poultry, including chickens. While chickens themselves may not show symptoms, they can carry the bacteria and spread it to humans through contaminated meat or eggs"]
        }
    mar_diagnosis = {
        'Coccidiosis' : ["रक्ती हगवण(Coccidiosis) हा जगभरातील कोंबड्यांना प्रभावित करणारा सर्वात सामान्य रोग आह जो कोंबडीच्या आतड्यांना नुकसान करतो.  Coccidiosis मधे वजन कमी होणे, खराब वाढ, अंडी उत्पादन कमी होणे आणि गंभीर प्रकरणांमध्ये मृत्यू देखील होऊ शकतो"],
        'Healthy' : [""],
        'NCD' : [""],
        'Salmonella' : [""]
    }
    
    class_symptoms = {
        'Coccidiosis': ["The signs of coccidiosis in chickens can vary depending on the severity of the infection and the species of Eimeria involved. Some common signs include:<br>1.Bloody diarrhea<br>2.Listlessness<br>3.Pale wattles and combs<br>4.Poor weight gain<br>5.Reduced egg production<br>5.Increased water consumption"],
        'NCD':["""Coughing and sneezing.
<br>Nasal discharge.
<br>Difficulty breathing.
<br>Diarrhea.
<br>Depression and lethargy.
<br>Drop in egg production.
"""],
        'Salmonella':["""Diarrhea with mucus or blood.
<br>Lethargy and decreased appetite.
<br>Ruffled feathers and drooping wings.
<br>Increased thirst and dehydration.
<br>Respiratory distress and coughing.
<br>Reduced egg production or abnormal eggs."""]
    }
    mar_symptoms = {
        'Coccidiosis' : ["पक्षी गर्दी करून कोपर्यात जाऊन बसतात<br>पंख पिंजार्ल्यासारखे दिसतात<br>डोळे मिटून गुंगून उभे राहतात<br>लालसर विटकरी रंगाचे विष्टा होते, गुदद्वाराचा भाग विष्टेने  भरतो<br>सिकाला (आतड्याचा भाग) सूज येते"],
        'Healthy' : [""],
        'NCD' : ["खूप ताप येतो<br>हिरवट, पिवळसर हगवण लागते<br>श्वास घेण्यास त्रास<br>नाकातून श्राव वाहतो<br>भूक मंदावते<br>मानमोडी झाल्यासारखे होते"],
        'Salmonella' : ["विशिष्ट लक्षणे दिसून येत नाहीत.<br>कोंबड्या घोळका करून उष्णतेच्या दिव्याच्या खाली एकत्र जमतात. आसपास गर्दी करतात<br>कोंबडीची तहान वाढते.<br>पाणी पिताना मागे तोल जाऊन कोंबडी पडते.<br>भूक मंदावते.<br>कोंबडी हिरव्या रंगाची पातळ संडास करतात, पंखे लुळे पडतात"]
    }

    class_firstAid = {
        'Coccidiosis': ["1.Isolate the sick chicken<br>2.Amprolium (in water) - Commonly used for mild cases.<br>3.Coccidiostats in feed"],
        'NCD':["""Isolate infected birds to prevent spread.
<br>Provide supportive care: fluids, warmth, and nutrition.
<br>Consult a veterinarian for specific treatment options.
<br>Vaccinate healthy birds to prevent infection.
<br>Implement strict biosecurity measures to prevent further spread.
<br>Monitor flock closely for any worsening symptoms.
 Medicines - MultiendServe Ginkgo biloba CH"""],
        'Salmonella':["""<br>Fluid therapy to combat dehydration.
<br>Isolation of infected birds to prevent spread.
<br>Clean and disinfect living areas and equipment.
<br>Adjust diet to support recovery and immune system.
<br>Monitor closely for any worsening symptoms.<br>Other medicines for Salmonella - neomycin, neomycin plus oxytetracycline, neomycin plus polymyxin, and sulfadiazine plus trimethoprim"""]
    }
    mar_firstAid = {
        'Coccidiosis' : ["आजारी कोंबडी अलग करा<br>एम्प्रोलियम (पाण्यात) - सामान्यतः सौम्य प्रकरणांसाठी वापरले जाते.<br>फीड मध्ये Coccidiostats"],
        'NCD' : ['''संसर्ग टाळण्यासाठी संक्रमित पक्ष्यांना वेगळे करा
                 <br>सहाय्यक काळजी प्रदान करा: द्रव, उबदारपणा आणि पोष
                 <br>विशिष्ट उपचार पर्यायांसाठी पशुवैद्याचा सल्ला घ्या.
                 <br>संसर्ग टाळण्यासाठी निरोगी पक्ष्यांना लस द्या.
                 <br>पुढील प्रसार टाळण्यासाठी कठोर जैवसुरक्षा उपाय लागू करा.
                 <br>कोणत्याही बिघडणाऱ्या लक्षणांसाठी कळपाचे बारकाईने निरीक्षण करा
                <br>औषधे - मल्टीएंडसर्व्ह जिन्कगो बिलोबा सीएच
'''],
        'Salmonella' : ['''निर्जलीकरण सोडविण्यासाठी द्रव थेरपी.
                        <br>प्रसार रोखण्यासाठी संक्रमित पक्ष्यांना वेगळे करणे
                        <br>राहण्याची जागा आणि उपकरणे स्वच्छ आणि निर्जंतुक करा.
                        <br>पुनर्प्राप्ती आणि रोगप्रतिकारक शक्तीला समर्थन देण्यासाठी आहार समायोजित करा
                        <br>कोणत्याही बिघडणाऱ्या लक्षणांसाठी बारकाईने निरीक्षण करा.
        ''']
    }

    biosecurity = """Limit Visitors: Only allow essential personnel near your birds. <br>
Footwear Protocol: Dedicate shoes or boots for coop use only. Disinfect them regularly. <br>
Equipment Sharing: Don't share feeders, waterers, or tools between flocks. Clean and disinfect after each use. <br>
Rodent & Pest Control: Seal cracks, eliminate hiding spots, and manage rodent populations to prevent disease spread. <br>
Wild Bird Discouragement: Use netting or deterrents to minimize contact with wild birds that can carry diseases. <br>
New Bird Quarantine: Isolate and monitor new birds before introducing them to your existing flock."""


    col5, col6 = st.columns(2)
    # Add content to the first column
    with col5:
        image = Image.open(file)
        st.image(image, width=300, caption="Uploaded Image")
        image_class, score_preds = upload_predict(file, model)

    # Add content to the second column
    with col6:
        if image_class in class_messages:
          st.subheader(class_messages[image_class])
          #st.audio(data=audio_file, autoplay=True)
          if(image_class != 'Healthy'):
            if(image_class =='Salmonella'):
                st.error('Isolate your chicken immediately!')
            st.info("Scroll down to know more")
          else:
              st.success("चांगले पोल्ट्री उपाय वापरत रहा !")
          

    if(image_class != 'Healthy'):
        if image_class in class_symptoms:
              col1, col2, col3, col4 = st.columns(4)
              with col1:
                  st.subheader("Diagnosis")
                  for diag in class_diagnosis[image_class]:
                    st.markdown(diag,unsafe_allow_html=True)
                  for diag in mar_diagnosis[image_class]:
                    st.markdown(diag,unsafe_allow_html=True)

              with col2:
                  st.subheader("Symptoms")
                  for symptom in class_symptoms[image_class]:
                    st.markdown(symptom,unsafe_allow_html=True)
                  for symptom in mar_symptoms[image_class]:
                    st.markdown(symptom,unsafe_allow_html=True)
              with col3:
                  st.subheader("First Aid")
                  for FirstAid in class_firstAid[image_class]:
                    st.markdown(FirstAid,unsafe_allow_html=True)
                  for FirstAid in mar_firstAid[image_class]:
                    st.markdown(FirstAid,unsafe_allow_html=True)

              with col4:
                  # Placeholder for additional content
                  st.subheader("Biosecurity Measures")
                  st.markdown(biosecurity,unsafe_allow_html=True)
              
              st.info("Still not sure? Get advice from Expert doctors from our [website](https://poultrymitra.netlify.app/doctors)!")
              

    
