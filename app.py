from predictor import endpoint, load_model
import streamlit as st
import numpy as np

# set page config
st.set_page_config(
	page_title="Analyze Your Apps Review",
	page_icon="📲"
)

# load model
with st.spinner("Loading our awesome AI 🤩. Please wait ..."):
	model = load_model()

@st.cache_data
def handle_text(text):
	# predict
	prediction = endpoint(text)
	my_array = np.array(prediction)
	# Convert to string
	result_str = my_array[0]

	# return
	return result_str
# title and subtitle
st.title("📲 Apps Review Sentiment Analysis")
st.write("Do you think that your customer loves your Apps? Do they love the facility you gave? 🛎️")
st.write("Checking all the review is not an easy task, let our AI do it for you! 😆")
st.write("It's easy and fast. Put the review down below and we will take care the rest 😉")

# user input
user_review = st.text_area(
	label="Review:",
	help="Input your (or your client's) review here, then click anywhere outside the box."
)

if user_review != "":
	prediction = handle_text(user_review)

	# display prediction
	st.subheader("AI thinks that ...")

	# check prediction
	if prediction == "positive":
		st.write(f"YAY! It's a positive review 🥰🥰.")
	elif prediction == "negative":
		st.write(f"NOO! It's a negative review 😱😱.")
	else:
		st.write(f"hmm its neutral review 😐😐. ")
