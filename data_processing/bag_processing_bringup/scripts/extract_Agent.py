#!/usr/bin/env python

import sys
import rospy
from people_msgs.msg import People, Person
import json
from std_msgs.msg import Header
import csv
import pandas as pd
from shapely.geometry import Polygon, Point

NODE_NAME = 'extract_agent'
NODE_RATE = 10 #Hz
MAP_BOUNDARIES = [(5.45, -4.66), (0.75, -0.56), (0.47, -0.73), (-0.73, 0.28), 
                  (0.01, 1.05), (-0.37, 1.37), (0.69, 2.62), (1.29, 2.14),
                  (2.01, 2.82), (2.95, 1.82), (3.52, 1.4), (4.16, 1.17), (7.86, -1.85)]
MAP = Polygon(MAP_BOUNDARIES)     

class AgentExtracter():
    """
    Class handling data
    """
    
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        
        self.H1 = None
        self.H2 = None
        
        with open(PEOPLE_ID) as json_file:
            peopleID = json.load(json_file)
            self.A_IDs = peopleID[BAG]
                
        # Person subscriber
        rospy.Subscriber('/people_tracker/people', People, self.cb_handle_data)
        
                                                    
                
    def cb_handle_data(self, people: People):
        """
        People callback

        Args:
            people (People): tracked people
        """
        self.H1 = None
        self.H2 = None
        for p in people.people:
            if self.H1 is None and int(p.name) in self.A_IDs["A1"]:
                # check if traj point is contained in the map 
                if not MAP.contains(Point(p.position.x, p.position.y)):
                    continue
                else:
                    self.H1 = p
                
            if self.H2 is None and int(p.name) in self.A_IDs["A2"]:
                if not MAP.contains(Point(p.position.x, p.position.y)):
                    continue
                else:
                    self.H2 = p
        

if __name__ == '__main__':
    BAG = sys.argv[1]
    PEOPLE_ID = sys.argv[2]
    CSV_PATH = sys.argv[3]

    # Init node
    rospy.init_node(NODE_NAME)

    # Set node rate
    rate = rospy.Rate(NODE_RATE)

    AE = AgentExtracter()

    pub = rospy.Publisher('/people_tracker/people_filtered', People, queue_size=10)

    # Open CSV file for writing
    with open(CSV_PATH + '/trajectory.csv', mode='w') as trajectory_file:
        trajectory_writer = csv.writer(trajectory_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        trajectory_writer.writerow(['Time Step', 'H1_X', 'H1_Y', 'H2_X', 'H2_Y'])

        while not rospy.is_shutdown():
            people_filtered_msg = People()
            people_filtered_msg.header = Header()
            people_filtered_msg.header.stamp = rospy.Time.now()
            people_filtered_msg.header.frame_id = 'map'

            H1 = Person()
            H1.name = "1"
            if AE.H1 is not None:
                H1.position = AE.H1.position
                h1_x = AE.H1.position.x
                h1_y = AE.H1.position.y
            else:
                H1.position.x = -3000
                H1.position.y = -3000
                h1_x = None
                h1_y = None

            H2 = Person()
            H2.name = "2"
            if AE.H2 is not None:
                H2.position = AE.H2.position
                h2_x = AE.H2.position.x
                h2_y = AE.H2.position.y
            else:
                H2.position.x = -3000
                H2.position.y = -3000
                h2_x = None
                h2_y = None

            people_filtered_msg.people.append(H1)
            people_filtered_msg.people.append(H2)

            pub.publish(people_filtered_msg)
            rate.sleep()

            # Write data to CSV
            trajectory_writer.writerow([rospy.Time.now().to_sec(), h1_x, h1_y, h2_x, h2_y])
            
    # Close CSV file after writing
    trajectory_file.close()
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(CSV_PATH + '/trajectory.csv')

    # Interpolate NaN values
    df.interpolate(method='linear', axis=0, inplace=True)
    df.bfill(axis=0, inplace=True)

    # Save the interpolated DataFrame back to CSV if needed
    df.to_csv(CSV_PATH + '/trajectory_interp.csv', index=False)