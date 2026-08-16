import os

def show_tree(folder, indent=""):
    items = sorted(os.listdir(folder))

    for item in items:
        path = os.path.join(folder, item)

        if os.path.isdir(path):
            print(indent + "[FOLDER] " + item)
            show_tree(path, indent + "    ")
        else:
            print(indent + item)


print("MARKET VOLATILITY PROJECT")
print("==========================")

show_tree(".")