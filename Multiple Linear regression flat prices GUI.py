import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tkinter import *
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('eladó lakások Szolnok.csv')
X = dataset.iloc[0: ,0:3].values
Y = dataset.iloc[0: ,-1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

lin_regressor = LinearRegression()
lin_regressor.fit(X_train, Y_train)

def model_pred():
    area1 = entry1.get()
    area = int(area1)
    
    room1 = entry2.get()
    room = int(room1)
    
    balcony1 = entry3.get()
    balcony = int(balcony1)
    
    tran_variables = np.array([[area, room, balcony]])
    pred_price = lin_regressor.predict(tran_variables).round(2)
    pred_price = str(pred_price)[1:-1]
    
    label1 = Label(window, text=(f'{area}, {room}, {balcony}  --->  Ár: {pred_price} mFt'), fg='green', font=("Verdana", 15))
    label1.pack()
    
    entry1.delete(0, END)
    entry2.delete(0, END)
    entry3.delete(0, END)


window = Tk()
window.geometry("600x600")
window.title("Szolnok lakás árak GUI")

# ---------------------------------------------------------------------------------------
label1 = Label(window, text = "Add meg a lakás alapterületét négyzetméterben", fg='red', font=("Verdana", 12))
label1.pack()
area = StringVar()
area.set("")
entry1 = Entry(window, textvariable=area, fg='green', width=10, font=("Verdana",12))
entry1.pack()


label2 = Label(window, text = "Add meg a szobák számát", fg='red', font=("Verdana", 12))
label2.pack()
room = StringVar()
room.set("")
entry2 = Entry(window, textvariable=room, fg='green', width=10, font=("Verdana",12))
entry2.pack()


label3 = Label(window, text = "Add meg az erkély területét négyzetméterben (ha nincs, akkor 0)", fg='red', font=("Verdana", 12))
label3.pack()
balcony = StringVar()
balcony.set("")
entry3 = Entry(window, textvariable=balcony, fg='green', width=10, font=("Verdana",12))
entry3.pack()
# ---------------------------------------------------------------------------------------


pred_button = Button(window, text="Predikció", fg='red', command=model_pred, height=2, width=15)
pred_button.pack()

mainloop()
    

