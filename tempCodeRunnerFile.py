
app.route('/data_page/')
def display_image():
    with open("fin_result.txt","r") as f5:
        result = f5.read()
    with open("rasa/data.json","w") as f1:
        json.dump({"text":[]},f1)
	#print('display_image filename: ' + filename)
    return render_template("data_page.html")