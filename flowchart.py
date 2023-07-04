from diagrams import Diagram, Edge
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.integration import SQS
from diagrams.generic import _Generic as Generic

with Diagram(name="LSTM Model Prediction", show=True):
    input_data = Generic("Input Data")
    lstm = EC2("LSTM Model")
    output_data = Generic("Output Data")
    prediction = Generic("Prediction")

    input_data >> lstm
    lstm >> output_data
    output_data - Edge(label="Predicted Value") >> prediction

    while_loop = Generic("While Loop")
    input_data_loop = Generic("Input Data")
    lstm_loop = EC2("LSTM Model")
    output_data_loop = Generic("Output Data")
    prediction_loop = Generic("Prediction")

    lstm_loop << Edge(label="Last n (144) Data Points") << input_data_loop
    lstm_loop >> output_data_loop
    output_data_loop - Edge(label="Predicted Value") >> prediction_loop
    while_loop << Edge(label="Loop until desired number of predictions have been made") << prediction_loop
    prediction - Edge(label="Add into") >> input_data_loop
