import cv2 as cv2
import face_recognition as fr
import os
import numpy

# Create database
route = 'Employees'
my_images = []
employee_names = []
employees_list = os.listdir(route)

for name in employees_list:
    current_image = cv2.imread(f'{route}\{name}')
    my_images.append(current_image)
    employee_names.append(os.path.splitext(name)[0])

print(employee_names)

# Encode images
def encode(images):

    # Create a new list
    encoded_list = []

    # Conver every image to rgb
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Encode
        encoded = fr.face_encodings(image)[0]

        # Add to the list
        encoded_list.append(encoded)

    # Return enconded list
    return encoded_list


encoded_employee_list = encode(my_images)

# Take a photo from the webcam
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Read image from the webcam
sucess, image = capture.read()

if not sucess:
    print("The capture could not be taken")
else:
    # Recognize face in the capture
    captured_face = fr.face_locations(image)

    # Encode captured face
    encoded_captured_face = fr.face_encodings(image, captured_face)

    # Find matches
    for encodface, locface in zip(encoded_captured_face, captured_face):
        matches = fr.compare_faces(encoded_employee_list, encodface)
        distances = fr.face_distance(encoded_employee_list, encodface)

        print(distances)

        match_index = numpy.argmin(distances)

        # Show matches if any
        if distances[match_index] > 0.6:
            print("There's no match with any of our employees.")
        else:
            print('Welcome to the office.')