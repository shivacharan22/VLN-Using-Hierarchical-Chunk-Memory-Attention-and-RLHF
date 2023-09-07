import streamlit as st
import pandas as pd
from streamlit_image_comparison import image_comparison

st.set_page_config("Vision language navigation Results Display", layout="wide")

st.header("Vision language navigation Results Display")

st.markdown("### sample Train data")

st.json({
    "distance": 13.42, 
    "scan": "ac26ZMwG7aT", 
    "path_id": 4928, 
    "path": [
      "47bec259eaac4e00a4db928751889af6", 
      "ecff9ecf0cfe4d8bb83260fc092f3b00", 
      "d1ffe5280fce4ac5a949cdc9ee8b6f7c", 
      "88fe76c2969d431caf5d60ff7aa5467a", 
      "6f8be003180e485aa6312a534469785b", 
      "d2bf18703fd645bcb077523c3c200c30", 
      "49f26e22f7514331b35d76bc1b8bb9f5"
    ], 
    "heading": 4.088, 
    "instructions": [
      "Exit the room using the door on the right. Turn left and go into the room on the left. Turn right and go into the room on the right. Turn left and wait in the hallway. ", 
      "Walk past the pink and blue patterned arm chair and through the doorway to the right. Turn a hard left and walk into the office and continue to the right past the bookshelves. Turn right through the doorway and then turn a hard left through the second doorway. Wait outside the door to the restroom. ", 
      "Go out the door behind the red chair. Go into the brown office room. Go into the bedroom. Go in front of the bathroom door. "
    ]
  },)

st.markdown("### Part1 : Bidirectional Encoder Representations from Transformers and universal sentence encoder with self-supervised learning and pre-training on navigation instructions and images")


tab1, tab2, tab3 = st.tabs(["1", "2", "3"])
colt1, colt2 = st.columns(2)
coltt1, coltt2 = st.columns(2)
colttt1, colttt2 = st.columns(2)
coli1, coli2, coli3 = st.columns(3)
coli21, coli22, coli23 = st.columns(3)
colii1, colii2, colii3 = st.columns(3)
colii21, colii22, colii23 = st.columns(3)
coliii1, coliii2, coliii3 = st.columns(3)
coliii21, coliii22, coliii23 = st.columns(3)

with tab1:
    st.header("Example ")
    with colt1:
        st.header("Sentences with the same meaning")
        st.markdown('##Sentence 1')
        st.markdown("**:red[Walk down one flight of stairs and stop on the landing]** ")
        st.markdown('##Sentence 2')
        st.markdown("**:blue[Walk between the columns and make a sharp turn right. Walk down the steps and stop on the landing. ]** ")
        with coli1:
            st.markdown('Model-1 :BERT Pretrain')
            st.metric("Similarity", "0.9809")
            st.metric("Eucildean", "3.2269")

        with coli2:
            st.markdown('Model-2 :BERT Pretrain+Self-Supervised')
            st.metric("Similarity", "0.9999")
            st.metric("Eucildean", "0.1161")
 
        with coli3:
            st.markdown('Model-3 :USE')
            st.metric("Similarity", "0.9511")
            st.metric("Eucildean", "4.6310")

    with colt2:
        st.header("Sentences with the different meaning")
        st.markdown('##Sentence 1')
        st.markdown("**:red[Walk down one flight of stairs and stop on the landing]** ")
        st.markdown('##Sentence 2')
        st.markdown("**:blue[Head straight until you pass the wall with holes in it the turn left and wait by the glass table with the white chairs. ]** ")
        with coli21:
            st.markdown('Model-1 :BERT Pretrain')
            st.metric("Similarity", "0.9550")
            st.metric("Eucildean", "4.9207")

        with coli22:
            st.markdown('Model-2 :BERT Pretrain+Self-Supervised')
            st.metric("Similarity", "0.9998")
            st.metric("Eucildean", "0.1535")
 
        with coli23:
            st.markdown('Model-3 :USE')
            st.metric("Similarity", "0.6069")
            st.metric("Eucildean", "4.6310")

with tab2:
    st.header("Example")
    with colt1:
        st.header("Sentences with the same meaning")
        st.markdown('##Sentence 1')
        st.markdown("**:red[Head straight until you pass the wall with holes in it the turn left and wait by the glass table with the white chairs]** ")
        st.markdown('##Sentence 2')
        st.markdown("**:blue[Go down past some room dividers toward a glasstop desk, and turn into the dining area. Wait next to the large glass dining table.]** ")
        with coli1:
            st.markdown('Model-1 :BERT Pretrain')
            st.metric("Similarity", "0.9647")
            st.metric("Eucildean", "4.3826")

        with coli2:
            st.markdown('Model-2 :BERT Pretrain+Self-Supervised')
            st.metric("Similarity", "0.9999")
            st.metric("Eucildean", "0.1033")
 
        with coli3:
            st.markdown('Model-3 :USE')
            st.metric("Similarity", "0.6831")
            st.metric("Eucildean", "5.0905995")

    with colt2:
        st.header("Sentences with the different meaning")
        st.markdown('##Sentence 1')
        st.markdown("**:red[Head straight until you pass the wall with holes in it the turn left and wait by the glass table with the white chairs. ]** ")
        st.markdown('##Sentence 2')
        st.markdown("**:blue[Exit the bedroom via the farthest left. Walk toward the couch. Stop there. ]** ")
        with colii21:
            st.markdown('Model-1 :BERT Pretrain')
            st.metric("Similarity", "0.9532")
            st.metric("Eucildean", "5.0486")

        with colii22:
            st.markdown('Model-2 :BERT Pretrain+Self-Supervised')
            st.metric("Similarity", "0.9999")
            st.metric("Eucildean", "0.1277")
 
        with colii23:
            st.markdown('Model-3 :USE')
            st.metric("Similarity", "0.5174")
            st.metric("Eucildean", "6.514755")
with tab3:
    st.header("Example ")
    with colt1:
        st.header("Sentences with the same meaning")
        st.markdown('##Sentence 1')
        st.markdown("**:red[Walk into the hallway and through the entrance to the kitchen area. Walk  Passed the sink and stove area and stop between the refrigerator and dining table.]** ")
        st.markdown('##Sentence 2')
        st.markdown("**:blue[Walk through the kitchen. Go past the sink and stove stand in front of the dining table on the bench side. ]** ")
        with coliii1:
            st.markdown('Model-1 :BERT Pretrain')
            st.metric("Similarity", "0.9875")
            st.metric("Eucildean", "2.6148")

        with coliii2:
            st.markdown('Model-2 :BERT Pretrain+Self-Supervised')
            st.metric("Similarity", "1.0000")
            st.metric("Eucildean", "0.0708")
 
        with coliii3:
            st.markdown('Model-3 :USE')
            st.metric("Similarity", "0.8160")
            st.metric("Eucildean", "3.9826872")

    with colt2:
        st.header("Sentences with the different meaning")
        st.markdown('##Sentence 1')
        st.markdown("**:red[Walk into the hallway and through the entrance to the kitchen area. Walk  Passed the sink and stove area and stop between the refrigerator and dining table.]** ")
        st.markdown('##Sentence 2')
        st.markdown("**:blue[Walk out of the bathroom and turn left. Walk along the hallway passed the white painting and the other bathroom.  Turn left towards the stairs. Walk down three of the stairs and wait on there. ]** ")
        with coliii21:
            st.markdown('Model-1 :BERT Pretrain')
            st.metric("Similarity", "0.9837")
            st.metric("Eucildean", "2.9844")

        with coliii22:
            st.markdown('Model-2 :BERT Pretrain+Self-Supervised')
            st.metric("Similarity", "0.9999")
            st.metric("Eucildean", "0.0901")
 
        with coliii23:
            st.markdown('Model-3 :USE')
            st.metric("Similarity", "0.7429")
            st.metric("Eucildean", "4.5409393")


st.markdown('## Images')

st.markdown('## Image similarity')
image_comparison(
    img1="/users/shiva/downloads/3.png",
    img2="/users/shiva/downloads/4.png"
)
st.metric(label="Cosine similarity", value="0.41")

st.markdown('## Image Dis - similarity')
image_comparison(
    img1="/users/shiva/downloads/3.png",
    img2="/users/shiva/downloads/b8cTxDM8gDG_2.png"
)
st.metric(label="Cosine similarity", value="0.23")


st.markdown("### Part2 : Whole network training with contrastive supervised learning")

df = pd.read_csv("/Users/shiva/rlhf/VLN-Using-Hierarchical-Chunk-Memory-Attention-and-RLHF/Part-2/conTra_data.csv")
st.markdown("## Dataset created")
st.dataframe(df.style.highlight_max(axis=0))

st.markdown('## Predicted by model 1 on train dataset sample')
video_file = open('/users/shiva/downloads/train_v1.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)

st.markdown('## Predicted by model 2 on train dataset sample')
video_file = open('/users/shiva/downloads/train_v2.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)

st.markdown('## Predicted by model 1 on unseen sample')
video_file = open('/users/shiva/downloads/val_v1.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)

st.markdown('## Predicted by model 2 on unseen sample')
video_file = open('/users/shiva/downloads/val_v2.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)




