import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
from tkinter import *


#------------------Testing the SVM-------------------------------------------
df = pd.read_csv('data.csv')
df.isnull().any()
df = df.fillna(method='ffill') #for NaN vals in data coz the program said there was
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)

clf = svm.SVC(kernel='linear', gamma=0.1, C=721)
#gamma https://www.youtube.com/watch?v=m2a2K4lprQw
	# low values - far
	# high values - close
# C https://www.youtube.com/watch?v=WVg5-vxQDm8
	# controls tradeoff between
	# smooth decision boundary and classifying training points correctly
	# large value of C more training points correct
	# less value of C smoother decision boundary

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)


#------------------Graph visualization---------------------------------------


# ------------------GUI creation----------------------------------------------
str = ""

class UIFrame:
	def __init__(self,master):
		master.resizable(False,False)

		self.canvas = Canvas(master, width="480", height="360")
		self.canvas.pack()

		self.background = PhotoImage(file="background.png")
		self.bglabel = Label(self.canvas, image=self.background)
		self.bglabel.place(x=0, y=0, relwidth=1, relheight=1)

		self.comval = StringVar()
		self.comment = Entry(self.canvas,justify="center",relief="flat",textvariable=self.comval)
		self.comment.config(font=("Calibri", 18), width="25")
		self.comm_w = self.canvas.create_window(240,160,anchor=CENTER,window=self.comment)

		self.button = Button(self.canvas, text="",relief="flat",command=self.btnClicked)
		self.btn_img = PhotoImage(file="button.png")
		self.button.config(font=("Calibri", 12), bg="#262b33", fg="#00ff00", image=self.btn_img)
		self.button_w = self.canvas.create_window(240,200,anchor=CENTER,window=self.button)

		self.resval = StringVar()
		self.result = Label(self.canvas,textvariable=self.resval)
		self.result.config(width=30, font=("Calibri", 18), bg="#262b33", fg="#ff00ff")  # ,bg="#1634cc"
		self.res_w = self.canvas.create_window(240,280,anchor=CENTER,window=self.result)

	def btnClicked(self):
		# print("Clicked")
		# print(username, password)
		# ------------------Processing Inputted Image File-----------------------------------
		# text = input("Enter a string:")
		filename = self.comment.get()
		text = process(text);

		# ------------------Predicting the Class--------------------------------------
		example_measures = np.array([text])
		example_measures = example_measures.reshape(len(example_measures), -1)
		prediction = clf.predict(example_measures)
		if prediction == [0]:
			#tm.showinfo("Sam and Pam Evaluation:", "The Verdict is Not Spam")
			self.resval.set("It's not spam!")
		else:
			#tm.showerror("Sam and Pam Evaluation:", "The Verdict is Spam")
			self.resval.set("It's spam.")


root = Tk()
mainprog = UIFrame(root)

root.mainloop()
