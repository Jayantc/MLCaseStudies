from sklearn.tree import DecisionTreeClassifier


def decisionTreeClassifier(weight, color):
    ball_features=[[35,1],[47,1],[90,0],[35,1],[92,0],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0],[48,1],[35,1],[35,1],[100,0]]
    names=[1,1,2,1,2,1,2,1,2,1,2,1,1,1,2]

    clf= DecisionTreeClassifier()
    clf.fit(ball_features, names)

    result= clf.predict([[weight,color]])

    if result==1:
        print('This is tennis ball')
    else:
        print('This is cricket ball')

def main():
    weight= int(input('Enter weight of your ball : '))
    color= input('Enter color of your ball yellow or red : ')
    if color=='yellow':
        color=1
    elif color=='red':
        color=0
    else:
        exit()
    decisionTreeClassifier(weight, color)

if __name__=='__main__':
    main()

