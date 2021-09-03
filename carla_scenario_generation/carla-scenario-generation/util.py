from shapely.geometry import LineString
import os
import glob

# The passerby area is the car area which is considered safe, when
# there is no one inside of it. In other words, the DANGER_AREA is
# an area used to make the decisions about driving, e.g., when a person
# intersects this area, a decicion needa be taken (brake or turn).
DANGER_AREA = [(640 - 255, 720), 
            (640 - 90, 720 - 250), 
            (640 + 90, 720 - 250), 
            (640 + 255, 720)]

# As the DANGER_AREA, the vehicle area works the same: it's an safe
# area, but, now, considering vehicles instead of people (or animals).
# It's a little bit lager than the danger area, but works as well.
WARNING_AREA = [(640 - 365, 720), 
                    (640 - 115, 720 - 250), 
                    (640 + 115, 720 - 250), 
                    (640 + 365, 720)]