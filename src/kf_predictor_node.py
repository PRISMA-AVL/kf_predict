#!/usr/bin/env python

# Import necessary modules
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from kalmanfilter_skim import *

# Create a list to hold the received PoseStamped messages
points = []
kf_pred_points = []
topic_to_read = '/natnet_ros/Robot_1/pose'
publish_frame = "world"
num_predictions = 50

# Function to publish the predicted path
def publish_prediction_path(prediction):
    global publish_frame
    # Create a Path message
    path_msg = Path()
    path_msg.header.stamp = rospy.Time.now()
    # Change frame for publishing
    path_msg.header.frame_id = publish_frame

    # Iterate over the arrays in the prediction
    for i in range(len(prediction[0])):
        # Create a PoseStamped message
        pose = PoseStamped()
        pose.pose.position.x = prediction[0][i]
        pose.pose.position.y = prediction[1][i]
        pose.pose.position.z = prediction[2][i]
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = publish_frame

        # Add the PoseStamped message to the Path message
        path_msg.poses.append(pose)

    # Publish the Path message
    path_publisher = rospy.Publisher('/prediction_path2', Path, queue_size=20)
    path_publisher.publish(path_msg)

# Function to predict the motion using Kalman Filters
def kf_predict(X0, motion_data, num_predictions):
    KF1, KF2, U1, U2 = initialize_kalman_filters(X0)
    KF1, KF2 = predict_motion(KF1, KF2, U1, U2, motion_data, num_predictions)
    return KF1

# Callback function to handle received PoseStamped data
def pose_stamped_callback(data):
    global num_predictions
    # Declare global variables
    global points, kf_pred_points

    # Log the position data from the PoseStamped message
    # rospy.loginfo("%s", data.pose.position)
    
    # Extract the x, y, z coordinates from the PoseStamped message and store them as a tuple
    kf_point = (data.pose.position.x, data.pose.position.y, data.pose.position.z)

    # Set the number of predictions to be made by the Kalman Filter

    # Append the new point to the list of points for the Kalman Filter prediction
    kf_pred_points.append(kf_point)

    # If the number of points exceeds the number of predictions, remove the oldest points
    if len(kf_pred_points) > num_predictions:
        kf_pred_points = kf_pred_points[-num_predictions:]

    # Print the current list of points for the Kalman Filter prediction
    # print("kf_pred_points:", kf_pred_points)

    # Extract the initial x, y, z coordinates from the first point in the list
    x_init = kf_pred_points[0][0]
    y_init = kf_pred_points[0][1]
    z_init = kf_pred_points[0][2]

    # Create the initial state vector for the Kalman Filter
    # The state vector contains the position and velocity (initialized to 0) of the object
    X0 = np.array([[x_init, y_init, z_init, 0, 0, 0]]).transpose()

    # Call the function to predict the motion using the Kalman Filter
    # The function returns the updated Kalman Filter after making the predictions
    KF1 = kf_predict(X0, kf_pred_points, num_predictions)

    # Extract the predicted x, y, z coordinates from the Kalman Filter
    prediction = (KF1.Xs[0,len(kf_pred_points):], KF1.Xs[1,len(kf_pred_points):], KF1.Xs[2,len(kf_pred_points):])

    # Call the function to publish the predicted path
    publish_prediction_path(prediction)

# Function to create a subscriber node
def subscriber_node():
    global topic_to_read
    # Initialize the ROS node
    rospy.init_node('subscriber_node', anonymous=True)

    # Create a subscriber for the PoseStamped topic
    rospy.Subscriber(topic_to_read, PoseStamped, pose_stamped_callback)

    # Spin the node to receive messages
    rospy.spin()

# Main function
if __name__ == '__main__':
    subscriber_node()
