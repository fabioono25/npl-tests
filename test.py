from transformers import pipeline

classifier = pipeline("text-classification")

# 20 example of tasks
tasks = [
"Go to the supermarket to buy some potatoes",
    "Having lunch with my wife",
    "Celebrating anniversary",
    "Completing my current ticket",
    "Paying the bills",
    "Going to the gym",
    "Going to the doctor",
    "Making dinner",
    "Buying tickets for the concert",
    "Exercise myself in the park",
]

for task in tasks:
  result = classifier(task)
  print(f"Task: {task}\nCategory: {result[0]['label']}\n")