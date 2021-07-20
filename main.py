from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import flask_monitoringdashboard as dashboard
import pandas as pd
import os
from flask_cors import CORS, cross_origin
from apps.training.train_model import TrainModel
from apps.prediction.predict_model import PredictModel
from apps.core.config import Config

app = Flask(__name__)
dashboard.bind(app)
CORS(app)

@app.route('/', methods=['POST','GET'])
def index_page():
 

    return render_template('index.html')



@app.route('/training', methods=['POST'])
@cross_origin()
def training_route_client():

    try:
        config = Config()
        #get run id
        run_id = config.get_run_id()
        data_path = config.training_data_path
        #trainmodel object initialization
        trainModel=TrainModel(run_id,data_path)
        #training the model
        trainModel.training_model()
        return Response("Training successfull! and its RunID is : "+str(run_id))
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)

@app.route('/batchprediction', methods=['POST'])
@cross_origin()
def batch_prediction_route_client():
 
    try:
        config = Config()
        #get run id
        run_id = config.get_run_id()
        data_path = config.prediction_data_path
        #prediction object initialization
        predictModel=PredictModel(run_id, data_path)
        #prediction the model
        predictModel.batch_predict_from_model()
        return Response("Prediction successfull! and its RunID is : "+str(run_id))
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.route('/prediction', methods=['POST'])
@cross_origin()
def single_prediction_route_client():
 
    try:
        config = Config()
        #get run id
        run_id = config.get_run_id()
        data_path = config.prediction_data_path
        print('Test')

        if request.method == 'POST':
            mean_radius = request.form['mean_radius']
            mean_texture = request.form["mean_texture"]
            mean_perimeter = request.form["mean_perimetert"]
            mean_area = request.form["mean_area"]
            mean_smoothness = request.form["mean_smoothness"]
           

            data = pd.DataFrame(data=[[mean_radius,mean_texture, mean_perimeter, mean_area,mean_smoothness]],
                              columns=['mean_radius','mean_texture', 'mean_perimeter', 'mean_area','mean_smoothness'])
            # using dictionary to convert specific columns
            convert_dict = {'mean_radius': int,
                            'mean_texture': float,
                            'mean_perimeter': float,
                            'mean_area': float,
                            'mean_smoothness': float
                            }

            data = data.astype(convert_dict)

            # object initialization
            predictModel = PredictModel(run_id, data_path)
            # prediction the model
            output = predictModel.single_predict_from_model(data)
            print('output : '+str(output))
            return Response("Predicted Output is : "+str(output))
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)

if __name__ == "__main__":
    #app.run()
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever() 
