from flask import Flask, render_template
import pickle
import pandas as pd
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

model = pickle.load(open('model.pkl', 'rb'))
data = pd.read_csv(r'data/Thecleverprogrammer.csv')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/plot.png', methods=['post'])
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def create_figure():
    predictions = model.predict(len(data), len(data) + 100)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(data['Views'])
    axis.plot(predictions)
    axis.legend(['Previous Data', 'Forecasted Data'])
    return fig


if __name__ == '__main__':
    app.run(debug=True)
