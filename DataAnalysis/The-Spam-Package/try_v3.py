import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
import matplotlib.pyplot as plt
import re
from tkinter import *
import tkinter.messagebox as tm

def process(text):
	textL = text.lower()
	chars = len(text)
	total = len(re.findall(r'\w+', text))
	if(total==0):
		total=1
	#spam words
	#count1 = textL.count("subscribe")/total
	count2 = textL.count(" sub ")/total
	count3 = textL.count("channel")/total
	count4 = textL.count("check out")/total
	count5 = textL.count("new")/total
	count6 = textL.count("www.")/total
	count7 = textL.count("need money")/total
	count8 = textL.count("my videos")/total
	count9 = textL.count("twitter")/total
	count10 = textL.count("follow")/total
	count11 = textL.count("https://")/total
	count12 = textL.count("checking")/total
	count13 = textL.count("please")/total
	#not spam words
	count14 = textL.count(" like ")/total
	count15 = textL.count("dislike")/total
	count16 = textL.count("why")/total
	count17 = textL.count("how")/total
	count18 = textL.count("view")/total
	count19 = textL.count("fuck")/total
	#characters count
	count20 = textL.count("!")/chars
	count21 = textL.count("?")/chars
	# I don't care about categorizing stuff anymore
	count22 = textL.count("#1")/total
	count23 = textL.count(".com")/total
	count24 = textL.count("comment")/total
	count25 = textL.count("check")/total
	count26 = textL.count("new")/total
	count27 = textL.count("yt")/total
	count28 = textL.count("raw")/total
	count29 = textL.count("buy")/total
	count30 = textL.count("iphone")/total
	count31 = textL.count("share")/total
	count32 = textL.count(".org")/total
	count33 = textL.count("cover")/total
	count34 = textL.count("minecraft")/total
	count35 = textL.count("loli")/total
	count36 = textL.count("money")/total
	count37 = textL.count("order")/total
	count38 = textL.count("download")/total
	count39 = textL.count("link")/total
	count40 = textL.count("android")/total
	count41 = textL.count("join")/total
	count42 = textL.count("pray")/total
	count43 = textL.count("trailer")/total
	count44 = textL.count("facebook")/total
	count45 = textL.count("mom")/total
	count46 = textL.count("blog")/total
	count47 = textL.count("fund")/total
	count48 = textL.count("vid ")/total
	count49 = textL.count("clip")/total
	count50 = textL.count("games")/total
	count51 = textL.count("app")/total
	count52 = textL.count("million")/total
	text = [count2, count3, count4,count5,count6,count7,count8,count9,count10,count11,count12,count13,count14,count15,count16,count17,count18,count19,count20,count21,count22,count23,count24,count25,count26,count27,count28,count29,count30,count31,count32,count33,count34,count35,count36,count37,count38,count39,count40,count41,count42,count43,count44,count45,count46,count47,count48,count49,count50,count51,count52]
	return text


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
# #figure number
# fignum = 1

# w = clf.coef_[0]
# a = -w[0]/w[1]
# xx = np.linspace(0,1)
# yy = a * xx - clf.intercept_[0]/w[1]
# h0 = plt.plot(xx, yy, 'k-', label='Hyperplane')

# # plot the parallels to the separating hyperplane that pass through the
# # support vectors (margin away from hyperplane in direction
# # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
# # 2-d.
# margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
# yy_down = yy - np.sqrt(1 + a ** 2) * margin
# yy_up = yy + np.sqrt(1 + a ** 2) * margin

# # plot the line, the points, and the nearest vectors to the plane
# plt.figure(fignum, figsize=(4, 3))
# plt.clf()
# plt.plot(xx, yy, 'k-')
# plt.plot(xx, yy_down, 'k--')
# plt.plot(xx, yy_up, 'k--')

# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#             facecolors='none', zorder=10, edgecolors='k')
# plt.scatter(X[:, 0], X[:, 1], c=y_train, zorder=10, cmap=plt.cm.Paired,
#             edgecolors='k')

# plt.axis('tight')
# x_min = -4.8
# x_max = 4.2
# y_min = -6
# y_max = 6

# XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
# Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(XX.shape)
# plt.figure(fignum, figsize=(4, 3))
# plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# plt.xticks(())
# plt.yticks(())
# fignum = fignum + 1

# plt.show()


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
		# ------------------Processing Input String-----------------------------------
		# text = input("Enter a string:")
		text = self.comment.get()
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
