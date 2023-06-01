# INSTALL LIBRARY YANG AKAN DIGUNAKAN DALAM PROJECT INI
# CHECK requirements.txt

# MENGIMPOR SEMUA LIBRARY YG AKAN DIGUNAKAN
import os
import cv2
import mysql.connector
import time
import numpy as np
import pandas as pd
from tkinter import *
from tkinter import ttk
from tkinter import messagebox as mb
from tkinter.font import Font
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from matplotlib.pyplot import text
from PIL import ImageTk, Image
from datetime import date
from datetime import datetime
from imutils import resize
from imutils.video import VideoStream

# MENGGUNAKAN ALGORITMA HAAR CASCADE CLASSIFIER UNTUK MENDETEKSI WAJAH
faceCascade = cv2.CascadeClassifier("face_detector/haarcascade_frontalface_default.xml")

lsDay = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
list_religion = ["Choose religion:", "Islam", "Christianity", "Catholicism", "Hinduism", "Buddhism", "Confuciaidm"]

# FUNGSI PENGECEKAN PATH
def check_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


# FUNGSI MEREKAM WAJAH SEBELUM MELAKUKAN REGISTER
def RegFace():
    try:
        getRegName = str(enRegName.get())
        getRegClass = str(enRegClass.get())
        getRegNumber = str(enRegNumber.get())
        getRegGender = str(enRegGender.get())
        getRegRelig = str(enRegRelig.get())
        getRegID = str(enRegID.get())
        getRegPW = str(enRegPW.get())
        
        mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
        mycursor = mysqldb.cursor()
        mycursor.execute("SELECT  * FROM student WHERE id=%s", (getRegID,))
        result = mycursor.fetchall()
        if len(result) == 0:
            check_path_exists("Training_Image/")        
            if (len(getRegName)>2 and len(getRegClass)>4 and len(getRegNumber)>0 and getRegRelig!="Choose religion:" and len(getRegID)>3 and len(getRegPW)>2):
                cam = cv2.VideoCapture(0)
                sampleNum = 0
                while (True):
                    ret, img = cam.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        sampleNum = sampleNum + 1
                        cv2.imwrite("Training_Image/" + str(getRegID) + "-" + getRegName + '-' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                        cv2.imshow('Taking Images', img)
                    if cv2.waitKey(50) & 0xFF == ord('q'):
                        break
                    elif sampleNum > 49:
                        break
                cam.release()
                cv2.destroyAllWindows()
                btnReg.place(x=930, y=550)
            else:
                mb.showwarning("Recheck Your Form", "Make sure the data entered is correct!")
        else:
            mb.showwarning("Registration Failed", "ID has been registered!")
    except:
        mb.showwarning("Ooops!", "Please fill data correctly!")


# FUNGSI MENGUBAH SEMUA GAMBAR DI path KE GRAYSCALE => PIL IMAGE => NUMPY ARRAY
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split("-")[0])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids


# FUNGSI MELAKUKAN TRAINING IMAGE
def TrainImages():
    check_path_exists("TrainingImageLabel/")
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces, ID = getImagesAndLabels("Training_Image")
    recognizer.train(faces, np.array(ID))
    recognizer.save("TrainingImageLabel\Trainner.yml")


# FUNGSI REGISTER AKUN PESERTA DIDIK
def register():
    getRegName = str(enRegName.get())
    getRegClass = str(enRegClass.get())
    getRegNumber = str(enRegNumber.get())
    getRegGender = str(enRegGender.get())
    getRegRelig = str(enRegRelig.get())
    getRegID = str(enRegID.get())
    getRegPW = str(enRegPW.get())

    try:
        mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
        mycursor = mysqldb.cursor()
        mycursor.execute("INSERT INTO student (id, name, class, number, gender, religion, password) VALUES (%s, %s, %s, %s, %s, %s, %s)", (getRegID, getRegName, getRegClass, getRegNumber, getRegGender, getRegRelig, getRegPW))
        mysqldb.commit()
        mysqldb.close()
    
        mb.showinfo("Registration success", "Your account has been successfully registered!\nPlease login")
        enRegName.delete(0, END)
        enRegClass.delete(0, END)
        enRegNumber.delete(0, END)
        enRegRelig.set(list_religion[0])
        enRegID.delete(0, END)
        enRegPW.delete(0, END)
        enUsername.focus_set()
        TrainImages()
        
    except Exception as e:
        mb.showwarning("Registration Failed!", "Please contact Administrator")


# FUNGSI PENGECEKAN MASKER
def checkMask(myID):
    datenow = date.today().strftime("%d-%m-%y")
    mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
    mycursor = mysqldb.cursor()
    mycursor.execute("SELECT  * FROM record WHERE id=%s AND date=%s", (myID, datenow,))
    result = mycursor.fetchall()
    mysqldb.close()
    if len(result) == 0:
        def detect_and_predict_mask(frame, faceNet, maskNet):
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

            faceNet.setInput(blob)
            detections = faceNet.forward()
            faces = []
            locs = []
            preds = []

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)

                    faces.append(face)
                    locs.append((startX, startY, endX, endY))
            if len(faces) > 0:
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)
            return (locs, preds)
        
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        maskNet = load_model("face_detector/mask_detector.model")

        vs = VideoStream(src=0).start()
        masker = 0

        while True:
            frame = vs.read()
            frame = resize(frame, width=600)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                if mask > withoutMask:
                    masker += 1    
                labEl = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if labEl == "Mask" else (0, 0, 255)
                labEl = "{}: {:.2f}%".format(labEl, max(mask, withoutMask) * 100)

                cv2.putText(frame, labEl, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.imshow("Mask Checker", frame)

            if (cv2.waitKey(1) == ord('q')):
                break
            elif (masker >= 15):
                time = datetime.now().strftime("%H:%M:%S")
                mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
                mycursor = mysqldb.cursor()
                mycursor.execute("INSERT INTO record (id, date, time) VALUES (%s, %s, %s)", (myID, datenow, time))
                mysqldb.commit()
                mysqldb.close()
                mb.showwarning("Attendance success", "Attendance successfully recorded")
                break            
        cv2.destroyAllWindows()
        vs.stop()
    else:
        mb.showwarning("Attendance failed", "Today, you have filled in attendance")
        enPassword.delete(0, END)


# FUNGSI TRACK/PENCOCOKKAN WAJAH SAAT ABSEN
def TrackImages():
    UID = "Unknown"
    poin = 0
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # exists3 = os.path.isfile("TrainingImageLabel\Trainner.yml")
    recognizer.read("TrainingImageLabel\Trainner.yml")
    
    cam1 = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, im = cam1.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                UID = str(serial)
            else:
                UID = "Unknown"
            cv2.putText(im, UID, (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('Taking Attendance', im)
        if (cv2.waitKey(1) == ord('q')):
            break
        elif (UID != "Unknown"):
            if poin > 20:
                cam1.release()
                cv2.destroyAllWindows()
                checkMask(UID)
                today = date.today()
                dates = today.strftime("%d-%m-%y")
                check_path_exists("image/proof/")
                cv2.imwrite("image/proof/" + UID + '_' + dates +".png", im)
                break
            poin += 1
        else:
            print("Object not detected")


# FUNGSI BERPINDAH FRAME
def raise_frame(frame):
    frame.tkraise()


root = Tk()
root.iconbitmap("image/icon.ico")
root.title("Face Recognition Attedance System And Mask Detection")
root.state('zoomed')
root.geometry("500x500")


# FUNGSI LOGIN PESERTA DIDIK
def Login():
    global data_user
    getLogUname = str(enUsername.get())
    getLogPW = str(enPassword.get())
    mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
    mycursor = mysqldb.cursor()
    mycursor.execute("SELECT  * FROM student WHERE id=%s AND password=%s", (getLogUname, getLogPW,))
    result = mycursor.fetchall()
    if len(result) > 0:
        data_user = result
        labDashID.config(text = data_user[0][0])
        labDashName.config(text = data_user[0][1])
        labDashClass.config(text = data_user[0][2])
        enMyName.insert(0, data_user[0][1])
        enMyClass.insert(0, data_user[0][2])
        enMyNumber.insert(0, data_user[0][3])
        enMyGender.set(data_user[0][4])
        enMyRelig.set(data_user[0][5])
        myID.insert(0, data_user[0][0])
        enMyPW.insert(0, data_user[0][6])
        raise_frame(dashboard)
        reqRecord()
    else:
        mb.showwarning("Login failed", "Wrong password!\nTry again")
        enPassword.delete(0, END)
    mysqldb.close()


# FUNGSI LOGIN ADMIN
def Logmin():
    getMinUname = str(enMinUname.get())
    getMinPW = str(enMinPW.get())
    mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
    mycursor = mysqldb.cursor()
    mycursor.execute("SELECT  * FROM admin WHERE id=%s AND password=%s", (getMinUname, getMinPW,))
    result = mycursor.fetchall()
    if len(result) > 0:
        raise_frame(adminpage)
        reqAllStudent()
    else:
        mb.showwarning("Login failed", "Wrong password!\nTry again")
        enPassword.delete(0, END)
    mysqldb.close()


logo80 = PhotoImage(file="X:/Python311/Face_Mask_Attendance_GUI/image/logo80.png")
logo60 = PhotoImage(file="X:/Python311/Face_Mask_Attendance_GUI/image/logo60.png")
avaPict = PhotoImage(file="X:/Python311/Face_Mask_Attendance_GUI/image/avatar.png")

adminpage = Frame(root, background="#FFF")
adminpage.place(x=0, y=0, width=1366, height=705)

logadpage = Frame(root, background="#01AAA3")
logadpage.place(x=0, y=0, width=1366, height=705)

dashboard = Frame(root, background="#FFF")
dashboard.place(x=0, y=0, width=1366, height=705)

homepage = Frame(root)
homepage.place(x=0, y=0, width=1366, height=705)

"""
HOME PAGE
"""
bghomeimg = PhotoImage(file="image/plawid.png")
bghome = Label(homepage, image=bghomeimg, bd=0)
bghome.place(x=0, y=0)

conLogoHome = Label(homepage, bg="#005aab", width=73, height=4)
conLogoHome.place(x=850, y=0)

homeLogo = Label(homepage, image=logo60, borderwidth = 0)
homeLogo.place(x=1105, y=3)

title = Label(homepage, text = "STUDENT ATTENDANCE AND MASK CHECKING SYSTEM", font=("Tahoma", 24), bg="#005aab", fg="#FFF", width=46, bd=0, padx=11, pady=14)
title.place(x=0, y=0)

conLogin = Label(homepage, bg="#CDD7D8", width=54, height=25, bd=0)
conLogin.place(x=236, y=200)

labLogin = Label(homepage, text = "Login to your account", font=("Tahoma Bold", 14), bg="#CDD7D8")
labLogin.place(x=280, y=250)

labUsername = Label(homepage, text = "Username:", font=("Tahoma", 14), bg="#CDD7D8")
labUsername.place(x=280, y=300)

enUsername = Entry(homepage, width = 26, font=12, bd=(2))
enUsername.place(x=280, y=340)

labPassword = Label(homepage, text = "Password:", font=("Tahoma", 14), bg="#CDD7D8")
labPassword.place(x=280, y=370)

enPassword = Entry(homepage, width = 26, font=12, bd=(2), show="❌")
enPassword.place(x=280, y=410)

# forget = Label(homepage, text = "Forget password", cursor= "hand2",  font=13, fg="#F00", justify=LEFT)

btnLogin = Button(homepage, text="Login", font=("Tahoma", 12),fg="#000", bg="#8fbbbf", command=Login)
btnLogin.place(x=280, y=470)

btnTakeAtt = Button(homepage, text="Take Attendance", font=("Tahoma", 12),fg="#FFF", bg="#3C595C", command=TrackImages)
btnTakeAtt.place(x=440, y=470)

btnNavLogad = Label(homepage, text="Login as Administrator", font=("Tahoma", 10, "underline"),fg="#000", bg="#cdd7d8")
btnNavLogad.place(x=480, y=550)
btnNavLogad.bind("<Button-1>", lambda e:raise_frame(logadpage))

"""
LOGIN ADMIN PAGE
"""
bglogadimg = PhotoImage(file="image/plawidmin.png")
bglogad = Label(logadpage, image=bglogadimg, bd=0)
bglogad.place(x=0, y=0)

conlogad = Label(logadpage, bg="#CDD7D8", width=54, height=25, bd=0)
conlogad.place(x=496, y=200)

labLogad = Label(logadpage, text = "Login to Admin Account", font=("Tahoma Bold", 14), bg="#CDD7D8")
labLogad.place(x=540, y=250)

labMinUname = Label(logadpage, text = "Username:", font=("Tahoma", 14), bg="#CDD7D8")
labMinUname.place(x=540, y=300)

enMinUname = Entry(logadpage, width = 26, font=12, bd=(2))
enMinUname.place(x=540, y=340)

labMinPW = Label(logadpage, text = "Password:", font=("Tahoma", 14), bg="#CDD7D8")
labMinPW.place(x=540, y=370)

enMinPW = Entry(logadpage, width = 26, font=12, bd=(2), show="❌")
enMinPW.place(x=540, y=410)

btnLogad = Button(logadpage, text="Login", font=("Tahoma", 12),fg="#000", bg="#8fbbbf", command=Logmin)
btnLogad.place(x=540, y=470)

btnNavLogstud = Label(logadpage, text="Login as Student", font=("Tahoma", 10, "underline"),fg="#000", bg="#cdd7d8")
btnNavLogstud.place(x=770, y=550)
btnNavLogstud.bind("<Button-1>", lambda e:raise_frame(homepage))

"""
REGISTER PAGE
"""

labHeadreg = Label(homepage, text = "ACCOUNT REGISTRATION FORM", font=("Tahoma", 22))
labHeadreg.place(x=890, y=110)

labRegName = Label(homepage, text = "Fullname:", font=("Poppins", 14))
labRegName.place(x=930, y=200)

enRegName = Entry(homepage, width = 30, font=("Poppins", 12))
enRegName.place(x=930, y=230)

labRegClass = Label(homepage, text = "Class:", font=("Poppins", 14))
labRegClass.place(x=930, y=260)

enRegClass = Entry(homepage, width = 10, font=("Poppins", 12), bd=(1))
enRegClass.place(x=930, y=290)

labRegNumber = Label(homepage, text = "Number:", font=("Poppins", 14))
labRegNumber.place(x=1127, y=260)

enRegNumber = Entry(homepage, width = 8, font=("Poppins", 12), bd=(1))
enRegNumber.place(x=1127, y=290)

labRegGender = Label(homepage, text = "Gender:", font=("Poppins", 14))
labRegGender.place(x=930, y=320)

enRegGender = StringVar()
enRegGender.set("M")

enRegGender1 = Radiobutton(homepage, text="Male", variable=enRegGender, value="M", font=("Poppins", 10))
enRegGender1.place(x=930, y=350)

enRegGender2 = Radiobutton(homepage, text="Female", variable=enRegGender, value="F", font=("Poppins", 10))
enRegGender2.place(x=1010, y=350)

labRegRelig = Label(homepage, text = "Religion:", font=("Poppins", 14))
labRegRelig.place(x=1127, y=320)

enRegRelig = StringVar()
enRegRelig.set(list_religion[0])

optFont = Font(family="Poppins", size=10)

optRelig = OptionMenu(homepage, enRegRelig, *list_religion)
optRelig.config(font=optFont)
optRelig.place(x=1127, y=350)

labRegID = Label(homepage, text = "Student ID:", font=("Poppins", 14))
labRegID.place(x=930, y=380)

enRegID = Entry(homepage, width = 11, font=("Poppins", 12), bd=(1))
enRegID.place(x=930, y=410)

labRegPW = Label(homepage, text = "Password:", font=("Poppins", 14))
labRegPW.place(x=930, y=440)

enRegPW = Entry(homepage, width = 20, font=12, bd=(1), show="*")
enRegPW.place(x=930, y=470)

btnRecface = Button(homepage, text="Record My Face", font=("Poppins", 13), bg="#005aab", fg="#FFF", command=RegFace)
btnRecface.place(x=930, y=510)

btnReg = Button(homepage, text="Register", font=("Poppins", 13), bg="#008000", fg="#FFF", command=register)

"""
DASHBOARD PESERTA DIDIK PAGE
"""
conNavDashboard = Label(dashboard, bg="#005aab", width=1366, height=5)
conNavDashboard.place(x=0, y=0)

conNavbar= Label(dashboard, bg="#CCC", width=1366, height=2)
conNavbar.place(x=0, y=82)

dashLogo = Label(dashboard, image=logo80, borderwidth = 0)
dashLogo.place(x=30, y=0)

dashProfPic = Label(dashboard, image=avaPict, borderwidth = 0)
dashProfPic.place(x=1030, y=5)

labDashID = Label(dashboard, text ="ID", font=("Tahoma", 10), fg="#FFF", background="#005aab")
labDashID.place(x=1120, y=10)

labDashName = Label(dashboard, text ="FULLNAME", font=("Tahoma", 10,), fg="#FFF", background="#005aab")
labDashName.place(x=1120, y=30)

labDashClass = Label(dashboard, text ="CLASS", font=("Tahoma", 10), fg="#FFF", background="#005aab")
labDashClass.place(x=1120, y=50)

labNavbarHome = Label(dashboard, text = "HOME    |", font=("Poppins", 11, "bold"), bg="#CCC")
labNavbarHome.place(x=35, y=90)
labNavbarHome.bind("<Button-1>", lambda e:raise_frame(homepage))

# FUNGSI MENGHAPUS SEMUA ISIAN ENTRY
def clear(event):
    enUsername.delete(0, END)
    enPassword.delete(0, END)
    enMinUname.delete(0, END)
    enMinPW.delete(0, END)
    
    enRegName.delete(0, END)
    enRegClass.delete(0, END)
    enRegNumber.delete(0, END)
    enMyRelig.set(list_religion[0])
    enRegRelig.set(list_religion[0])
    enRegGender.set("M")
    enRegID.delete(0, END)
    enRegPW.delete(0, END)
    
    enMyName.delete(0, END)
    enMyClass.delete(0, END)
    enMyNumber.delete(0, END)
    enMyRelig.set(list_religion[0])
    myID.delete(0, END)
    enMyPW.delete(0, END)
    enMyRePW.delete(0, END)
    enFilter.delete(0, END)
    enSearch.delete(0, END)
    enMinSearch.delete(0, END)
    
    dashTree.delete(*dashTree.get_children())
    minTree.delete(*minTree.get_children())

    labDashID.config(text = "ID")
    labDashName.config(text = "FULLNAME")
    labDashClass.config(text = "CLASS")
    
    enMinMyName.delete(0, END)
    enMinMyClass.delete(0, END)
    enMinMyNumber.delete(0, END)
    enMinMyGender.set("M")
    enMinMyRelig.set(list_religion[0])
    enMinMyID.configure(state='normal')
    enMinMyID.delete(0, END)
    
    data_user = []
    raise_frame(homepage)


navbarLogout = Label(dashboard, text = "LOGOUT", font=("Poppins", 11, "bold"), bg="#CCC")
navbarLogout.place(x=115, y=90)
navbarLogout.bind("<Button-1>", clear)

# FUNGSI MEMBUAT JAM DIGITAL
def tick():
    now = time.localtime().tm_wday
    day = lsDay[now]
    datenow = date.today().strftime("%d %B %Y")
    times = time.strftime('%H:%M:%S')
    time_string = day + ", " +  datenow + " | " + times
    navbarClock.config(text=time_string)
    navbarClock.after(200,tick)


labDashHead1 = Label(dashboard, text="STUDENT ATTENDANCE RECAPITULATION", font=("Poppins", 18), fg="#FFF", bg="#000", width=60, pady=7)
labDashHead1.place(x=0, y=119)

labDashHead2 = Label(dashboard, text="MY ACCOUNT", font=("Poppins", 18), fg="#FFF", bg="#7C99AC", width=40, pady=7)
labDashHead2.place(x=810, y=119)


# FUNGSI MENAMPILKAN BUKTI ABSEN
def showProof(event):
    slctRow = dashTree.focus()
    slctVal = dashTree.item(slctRow).get("values")[3]
    nwin = Toplevel()
    nwin.iconbitmap("image/icon.ico")
    nwin.title("PROOF "+ slctVal)
    picpath = "image/proof/" + str(data_user[0][0]) + "_" + slctVal + ".png"
    proofPic = ImageTk.PhotoImage(Image.open(picpath))
    proof = Label(nwin, image = proofPic)
    proof.pack()
    nwin.mainloop()


style = ttk.Style()
style.configure("Treeview", highlightthickness=0, bd=0, font=('Poppins', 11))
style.configure("Treeview.Heading", font=('Poppins', 11,'bold'))

dashTree_frame = Frame(dashboard, width=660, height=410, bg="#FFF")
dashTree_frame.place(x=75, y=200)

dashTree = ttk.Treeview(dashTree_frame, height =19, column=("c1", "c2", "c3", "c4", "c5"), show='headings')
dashTree.column("#1", anchor=CENTER, width=60)
dashTree.heading("#1", text="NO")
dashTree.column("#2", anchor=CENTER, width=150)
dashTree.heading("#2", text="ID")
dashTree.column("#3", anchor=CENTER, width=150)
dashTree.heading("#3", text="DAY")
dashTree.column("#4", anchor=CENTER, width=150)
dashTree.heading("#4", text="DATE")
dashTree.column("#5", anchor=CENTER, width=150)
dashTree.heading("#5", text="TIME")

dashTree.place(x=0, y=0)

# FUNGSI REQUEST ABSEN KE TREEVIEW (TABLE)
def reqRecord():
    dashTree.delete(*dashTree.get_children())    
    enFilter.delete(0, END)
    enSearch.delete(0, END)    
    mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
    mycursor = mysqldb.cursor()
    mycursor.execute("SELECT  * FROM record WHERE id=%s", (data_user[0][0],))
    result = mycursor.fetchall()

    num = 1
    for row in result:
        row = list(row)
        hari = lsDay[datetime.strptime(row[1], '%d-%m-%y').weekday()]
        row.insert(0, num)
        row.insert(2, hari)
        if num%2 == 0:
            dashTree.insert("", END, values=row, tags=("even",))
        else:
            dashTree.insert("", END, values=row, tags=("odd",))
        num += 1        
    mysqldb.close()


dashTree.tag_configure("even", foreground="black", background="white")
dashTree.tag_configure("odd", foreground="white", background="black")

dashTreescroll = Scrollbar(dashTree_frame, orient="vertical", command=dashTree.yview)
dashTree.configure(yscrollcommand=dashTreescroll.set)
dashTreescroll.place(x=660, y=0, relheight=1, anchor='ne')

dashTree.bind("<Double-Button-1>", showProof)

# FUNGSI FILTER ABSEN BERDASARKAN HARI
def filter():
    findDay = enFilter.get().capitalize()    
    mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
    mycursor = mysqldb.cursor()
    mycursor.execute("SELECT  * FROM record WHERE id=%s", (data_user[0][0],))
    result = mycursor.fetchall()
    
    dashTree.delete(*dashTree.get_children())
    
    if len(result) > 0:
        num = 1
        for row in result:
            row = list(row)
            day = lsDay[datetime.strptime(row[1], '%d-%m-%y').weekday()]
            if day == findDay:
                row.insert(0, num)
                row.insert(2, day)
                if num%2 == 0:
                    dashTree.insert("", END, values=row, tags=("even",))
                else:
                    dashTree.insert("", END, values=row, tags=("odd",))
                num += 1            
    mysqldb.close()
    

# FUNGSI MENCARI ABSEN BERDASARKAN TANGGAL    
def search():
    findDate = enSearch.get()    
    mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
    mycursor = mysqldb.cursor()
    mycursor.execute("SELECT  * FROM record WHERE id=%s AND date=%s", (data_user[0][0], findDate,))
    result = mycursor.fetchall()
    
    dashTree.delete(*dashTree.get_children())
    
    if len(result) > 0:
        num = 1
        for row in result:
            row = list(row)
            hari = lsDay[datetime.strptime(row[1], '%d-%m-%y').weekday()]
            row.insert(0, num)
            row.insert(2, hari)
            if num%2 == 0:
                dashTree.insert("", END, values=row, tags=("even",))
            else:
                dashTree.insert("", END, values=row, tags=("odd",))
            num += 1            
    mysqldb.close()


labFilter = Label(dashboard, text = "Filter Days   :", font=("Poppins", 11), bg="#FFF")
labFilter.place(x=75, y=620)

labSearch = Label(dashboard, text = "Search Date   :", font=("Poppins", 11), bg="#FFF")
labSearch.place(x=325, y=620)

enFilter = Entry(dashboard, width = 10, font=("Poppins", 12), bd=3, relief=GROOVE)
enFilter.place(x=170, y=620)

btnFilter = Button(dashboard, text="Filter", command=filter)
btnFilter.place(x=170, y=650)

enSearch = Entry(dashboard, width = 10, font=("Poppins", 12), bd=3, relief=GROOVE)
enSearch.place(x=435, y=620)

btnSearch = Button(dashboard, text="Search", command=search)
btnSearch.place(x=435, y=650)

btnSync = Button(dashboard, text="\U0001F501", font=("Poppins", 20, "bold"), bg="#FFF", bd=0, command=reqRecord)
btnSync.place(x=695, y=610)

"""
MY ACCOUNT
"""
conAcc= Label(dashboard, bg="#f0f0f0", width=80, height=45)
conAcc.place(x=810, y=165)

labMyName = Label(dashboard, text = "Fullname:", font=('Poppins', 12), bg="#f0f0f0")
labMyName.place(x=830, y=200)

enMyName = Entry(dashboard, width = 30, font=("Poppins", 12))
enMyName.place(x=990, y=203)

labMyClass = Label(dashboard, text = "Class:", font=('Poppins', 12), bg="#f0f0f0")
labMyClass.place(x=830, y=235)

enMyClass = Entry(dashboard, width = 10, font=("Poppins", 12), bd=(1))
enMyClass.place(x=990, y=238)

labMyNumber = Label(dashboard, text = "Number:", font=('Poppins', 12), bg="#f0f0f0")
labMyNumber.place(x=830, y=270)

enMyNumber = Entry(dashboard, width = 8, font=("Poppins", 12), bd=(1))
enMyNumber.place(x=990, y=275)

labMyGender = Label(dashboard, text = "Gender:", font=('Poppins', 12), bg="#f0f0f0")
labMyGender.place(x=830, y=305)

enMyGender = StringVar()
enMyGender.set("M")

enMyGender1 = Radiobutton(dashboard, text="Male", variable=enMyGender, value="M", font=("Poppins", 10))
enMyGender1.place(x=990, y=308)

enMyGender2 = Radiobutton(dashboard, text="Female", variable=enMyGender, value="F", font=("Poppins", 10))
enMyGender2.place(x=1070, y=308)

labMyReligion = Label(dashboard, text = "Religion:", font=('Poppins', 12), bg="#f0f0f0")
labMyReligion.place(x=830, y=340)

enMyRelig = StringVar()
enMyRelig.set(list_religion[0])

optFont = Font(family="Poppins", size=10)

optMyReligion = OptionMenu(dashboard, enMyRelig, *list_religion)
optMyReligion.config(font=optFont)
optMyReligion.place(x=990, y=340)

labMyID = Label(dashboard, text = "ID:", font=('Poppins', 12), bg="#f0f0f0")
labMyID.place(x=830, y=375)

myID = Entry(dashboard, width = 10, font=("Poppins", 12), bd=(1))
myID.place(x=990, y=378)

labMyPassword = Label(dashboard, text = "Password:", font=('Poppins', 12), bg="#f0f0f0")
labMyPassword.place(x=830, y=410)

enMyPW = Entry(dashboard, width = 20, font=12, bd=(1), show="*")
enMyPW.place(x=990, y=413)

labMyRePW = Label(dashboard, text = "Re-password:", font=('Poppins', 12), bg="#f0f0f0")
labMyRePW.place(x=830, y=445)

enMyRePW = Entry(dashboard, width = 20, font=12, bd=(1), show="*")
enMyRePW.place(x=990, y=448)


# FUNGSI MENG-UPDATE AKUN PESERTA DIDIK
def updateData():
    getMyPassword = enMyPW.get()
    getMyRepassword = enMyRePW.get()
    if getMyPassword == getMyRepassword:
        getMyName = enMyName.get()
        getMyClass = enMyClass.get()
        getMyNumber = str(enMyNumber.get())
        getMyGender = enMyGender.get()
        getMyRelig = enMyRelig.get()
        id = data_user[0][0]
        
        mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
        mycursor = mysqldb.cursor()
        # try:
        upquery = "UPDATE student SET id=%s, name=%s, class=%s, number=%s, gender=%s, religion=%s, password=%s WHERE id=%s"
        val = (id, getMyName, getMyClass, getMyNumber, getMyGender, getMyRelig, getMyRepassword, id)
        
        mycursor.execute(upquery, val)
        mysqldb.commit()
        mb.showinfo("Update success", "Your account has been updated successfully!")
        mysqldb.close()
        clear(id)
        raise_frame(homepage)
    else:
        mb.showinfo("Update failed", "Please fill in all data CORRECTLY!")


btnUpdate = Button(dashboard, text="Update Data", font=('Poppins', 11,'bold'), command=updateData)
btnUpdate.place(x=1100, y=490)

navbarClock = Label(dashboard, font=("Poppins", 10, "bold"), bg="#005aab" , fg="#FFF", pady=5, padx=5)
navbarClock.place(x=1135, y=675)
tick()

"""
ADMIN PAGE
"""
conNavminpage = Label(adminpage, bg="#005aab", width=1366, height=5)
conNavminpage.place(x=0, y=0)

conNavminbar= Label(adminpage, bg="#CCC", width=1366, height=2)
conNavminbar.place(x=0, y=82)

dashminlogo = Label(adminpage, image=logo80, borderwidth = 0)
dashminlogo.place(x=30, y=0)

labNavminBarHome = Label(adminpage, text = "HOME    |", font=("Poppins", 11, "bold"), bg="#CCC")
labNavminBarHome.place(x=35, y=90)
labNavminBarHome.bind("<Button-1>", lambda e:raise_frame(logadpage))

labNavminBarLogout = Label(adminpage, text = "LOGOUT", font=("Poppins", 11, "bold"), bg="#CCC")
labNavminBarLogout.place(x=115, y=90)
labNavminBarLogout.bind("<Button-1>", clear)

labDashminHead1 = Label(adminpage, text="STUDENT ACCOUNT", font=("Poppins", 18), fg="#FFF", bg="#000", width=100, pady=7)
labDashminHead1.place(x=0, y=119)

minTree_frame = Frame(adminpage, width=930, height=410, bg="#FFF")
minTree_frame.place(x=25, y=200)

minTree = ttk.Treeview(minTree_frame, height =19, column=("c1", "c2", "c3", "c4", "c5", "c6", "c7"), show='headings')
minTree.column("#1", anchor=CENTER, width=60)
minTree.heading("#1", text="NO")
minTree.column("#2", anchor=CENTER, width=100)
minTree.heading("#2", text="ID")
minTree.column("#3", anchor=CENTER, width=300)
minTree.heading("#3", text="NAME")
minTree.column("#4", anchor=CENTER, width=100)
minTree.heading("#4", text="CLASS")
minTree.column("#5", anchor=CENTER, width=100)
minTree.heading("#5", text="NUMBER")
minTree.column("#6", anchor=CENTER, width=100)
minTree.heading("#6", text="GENDER")
minTree.column("#7", anchor=CENTER, width=150)
minTree.heading("#7", text="RELIGION")

minTree.place(x=0, y=0)

#FUNGSI REQUEST DATA AKUN PESERTA DIDIK
def reqAllStudent():
    minTree.delete(*minTree.get_children())
    enMinSearch.delete(0, END)    
    enMinMyName.delete(0, END)
    enMinMyClass.delete(0, END)
    enMinMyNumber.delete(0, END)
    enMinMyGender.set("M")
    enMinMyRelig.set(list_religion[0])
    enMinMyID.configure(state='normal')
    enMinMyID.delete(0, END)
    
    mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
    mycursor = mysqldb.cursor()
    mycursor.execute("SELECT * FROM student")
    result = mycursor.fetchall()

    num = 1
    for row in result:
        row = list(row)
        row.insert(0, num)
        if num%2 == 0:
            minTree.insert("", END, values=row, tags=("even",))
        else:
            minTree.insert("", END, values=row, tags=("odd",))
        num += 1        
    mysqldb.close()


# FUNGSI MENDAPATKAN DATA DARI PILIHAN AKUN PESERTA DIDIK
def getDataSelected(event):
    enMinMyID.configure(state='normal')
    slctRow = minTree.focus()
    slctVal = minTree.item(slctRow).get("values")
    print(slctVal)
    enMinMyName.delete(0, END)
    enMinMyClass.delete(0, END)
    enMinMyNumber.delete(0, END)
    enMinMyGender.set("M")
    enMinMyRelig.set(list_religion[0])
    enMinMyID.delete(0, END)
    enMinMyName.insert(0, slctVal[2])
    enMinMyClass.insert(0, slctVal[3])
    enMinMyNumber.insert(0, slctVal[4])
    enMinMyGender.set(slctVal[5])
    enMinMyRelig.set(slctVal[6])
    enMinMyID.insert(0, slctVal[1])
    enMinMyID.configure(state='readonly')


minTree.tag_configure("even", foreground="black", background="white")
minTree.tag_configure("odd", foreground="white", background="black")

minTree.bind("<Double-Button-1>", getDataSelected)

minTreescroll = Scrollbar(minTree_frame, orient="vertical", command=minTree.yview)
minTree.configure(yscrollcommand=minTreescroll.set)
minTreescroll.place(x=930, y=0, relheight=1, anchor='ne')

# FUNGSI MENCARI NIS PESERTA DIDIK
def searchID():
    find = enMinSearch.get()    
    mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
    mycursor = mysqldb.cursor()
    mycursor.execute("SELECT  * FROM student WHERE id=%s", (find,))
    result = mycursor.fetchall()

    minTree.delete(*minTree.get_children())
    
    if len(result) > 0:
        num = 1
        for row in result:
            row = list(row)
            row.insert(0, num)
            if num%2 == 0:
                minTree.insert("", END, values=row, tags=("even",))
            else:
                minTree.insert("", END, values=row, tags=("odd",))
            num += 1                    
    mysqldb.close()


labMinSearch = Label(adminpage, text = "Find ID   :", font=("Poppins", 11), bg="#FFF")
labMinSearch.place(x=25, y=620)

enMinSearch = Entry(adminpage, width = 10, font=("Poppins", 12), bd=3, relief=GROOVE)
enMinSearch.place(x=135, y=620)

btnMinSearch = Button(adminpage, text="Search", command=searchID)
btnMinSearch.place(x=135, y=650)

btnMinSync = Button(adminpage, text="\U0001F501", font=("Poppins", 20, "bold"), bg="#FFF", bd=0, command=reqAllStudent)
btnMinSync.place(x=900, y=610)

conMinAcc = Label(adminpage, bg="#f0f0f0", width=80, height=45)
conMinAcc.place(x=960, y=165)

labMinMyName = Label(adminpage, text = "Fullname", font=('Poppins', 12), bg="#f0f0f0")
labMinMyName.place(x=970, y=200)

enMinMyName = Entry(adminpage, width = 30, font=("Poppins", 12))
enMinMyName.place(x=1090, y=203)

labMinmyClass = Label(adminpage, text = "Class", font=('Poppins', 12), bg="#f0f0f0")
labMinmyClass.place(x=970, y=235)

enMinMyClass = Entry(adminpage, width = 10, font=("Poppins", 12), bd=(1))
enMinMyClass.place(x=1090, y=238)

labMinMyNumber = Label(adminpage, text = "Number", font=('Poppins', 12), bg="#f0f0f0")
labMinMyNumber.place(x=970, y=270)

enMinMyNumber = Entry(adminpage, width = 8, font=("Poppins", 12), bd=(1))
enMinMyNumber.place(x=1090, y=275)

labMinMyGender = Label(adminpage, text = "Gender", font=('Poppins', 12), bg="#f0f0f0")
labMinMyGender.place(x=970, y=305)

enMinMyGender = StringVar()
enMinMyGender.set("M")

enMinMyGender1 = Radiobutton(adminpage, text="Male", variable=enMinMyGender, value="M", font=("Poppins", 10))
enMinMyGender1.place(x=1090, y=308)

enMinMyGender2 = Radiobutton(adminpage, text="Female", variable=enMinMyGender, value="F", font=("Poppins", 10))
enMinMyGender2.place(x=1200, y=308)

labMinmyRelig = Label(adminpage, text = "Religion", font=('Poppins', 12), bg="#f0f0f0")
labMinmyRelig.place(x=970, y=340)

enMinMyRelig = StringVar()
enMinMyRelig.set(list_religion[0])

optMinFont = Font(family="Poppins", size=10)

optMinMyRelig = OptionMenu(adminpage, enMinMyRelig, *list_religion)
optMinMyRelig.config(font=optMinFont)
optMinMyRelig.place(x=1090, y=340)

labMinMyID = Label(adminpage, text = "ID", font=('Poppins', 12), bg="#f0f0f0")
labMinMyID.place(x=970, y=375)

enMinMyID = Entry(adminpage, width = 10, font=("Poppins", 12), bd=(1))
enMinMyID.place(x=1090, y=378)


# FUNGSI MENGHAPUS AKUN PESERTA DIDIK
def deleteDdp():
    id = enMinMyID.get()
    
    confirmup = mb.askquestion("Delete Data Confirmation", "Do you wanna Delete this Account?", icon='warning')
    if confirmup == 'yes':
        mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
        mycursor = mysqldb.cursor()
        upquery = "DELETE from student WHERE id=%s"
        val = (id,)
        
        mycursor.execute(upquery, val)
        mysqldb.commit()
        mysqldb.close()
        reqAllStudent()
    else:
        print("Not Deleted")


# FUNGSI MEN-UPDATE AKUN PESERTA DIDIK
def updateData():
    getMyName = enMinMyName.get()
    getMyClass = enMinMyClass.get()
    getMyNumber = str(enMinMyNumber.get())
    getMyGender = enMinMyGender.get()
    getMyRelig = enMinMyRelig.get()
    id = enMinMyID.get()
    
    confirmup = mb.askquestion("Update Data Confirmation", "Do you wanna Update this Account?", icon='warning')
    if confirmup == 'yes':
        mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="attendance")
        mycursor = mysqldb.cursor()

        upquery = "UPDATE student SET name=%s, class=%s, number=%s, gender=%s, religion=%s WHERE id=%s"
        val = (getMyName, getMyClass, getMyNumber, getMyGender, getMyRelig, id)            
        mycursor.execute(upquery, val)
        mysqldb.commit()
        mysqldb.close()
    else:
        print("Not Updated")    
    reqAllStudent()

btnDel = Button(adminpage, text="DELETE", font=("Poppins", 13), bg="#ff373d", fg="#FFF", command=deleteDdp)
btnDel.place(x=1150, y=650)

btnUpd = Button(adminpage, text="UPDATE",font=("Poppins", 13), bg="#005aab", fg="#FFF", command=updateData)
btnUpd.place(x=1250, y=650)

root.mainloop()