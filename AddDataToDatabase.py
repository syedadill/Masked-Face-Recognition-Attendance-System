import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://realtimefaceattendance-26fa3-default-rtdb.firebaseio.com/"
})

ref=db.reference('Students')

data = {
    "08032":
        {
            "name": "Dr. Asad Ahmed",
            "major": "Machine Learning",
            "starting year": 2019,
            "total_attendance": 7,
            "standings": "G",
            "year": 4,
            "last_attendance_time": "2023-06-3 00:05:35"
        },
    "23423":
        {
            "name": "Syed Adil Gillani",
            "major": "Deep Learning",
            "starting year": 2019,
            "total_attendance": 12,
            "standings": "G",
            "year": 4,
            "last_attendance_time": "2023-08-3 00:05:35"
        },
    "32324":
        {
            "name": "Ghulam Murtaza",
            "major": "Ethical Hacking",
            "starting year": 2019,
            "total_attendance": 12,
            "standings": "Marvellous",
            "year": 4,
            "last_attendance_time": "2023-10-03 00:05:35"
        },
}
for key, value in data.items():
    ref.child(key).set(value)

