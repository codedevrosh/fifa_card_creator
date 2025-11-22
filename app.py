import streamlit as st
import numpy as np
import pickle
import random
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import textwrap
# ------------------------------
# load models & scaler (pickles)
# ------------------------------
role_model = pickle.load(open("classification_pipeline.pkl", "rb"))
overall_model = pickle.load(open("regression_pipeline.pkl", "rb"))
# ------------------------------
# club logos (example CDN links)
# Replace with your own local logos if you have them.
# ------------------------------
club_logo_map = {
    "Real Madrid": "https://upload.wikimedia.org/wikipedia/en/5/56/Real_Madrid_CF.svg",
    "Barcelona": "https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg",
    "Manchester United": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "PSG": "https://upload.wikimedia.org/wikipedia/en/a/a7/Paris_Saint-Germain_F.C..svg",
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "Bayern Munich": "data:image/png;base64,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",
    "Juventus": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Juventus_FC_-_logo_black_%28Italy%2C_2020%29.svg/195px-Juventus_FC_-_logo_black_%28Italy%2C_2020%29.svg.png",
    "Inter Milan": "data:image/png;base64,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",
    "AC Milan": "https://upload.wikimedia.org/wikipedia/commons/d/d0/Logo_of_AC_Milan.svg"
}
# default club list
clubs = list(club_logo_map.keys())
# role mapping (must match your cluster encoding)
role_map = {0: "Midfielder", 1: "Goalkeeper", 2: "Attacker", 3: "Defender"}
# ------------------------------
# Utility: fetch image from URL (returns PIL)
# ------------------------------
def fetch_image(url, size=(100,100)):
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGBA")
        img.thumbnail(size, Image.Resampling.LANCZOS)   # FIX
        return img
    except Exception:
        im = Image.new("RGBA", size, (200,200,200,255))
        return im

# ------------------------------
# Utility: compute rarity from overall
# ------------------------------
def compute_rarity(overall):
    if overall >= 90:
        return "Icon", "#ffd700"   # shiny gold
    if overall >= 80:
        return "Gold", "#e6c400"
    if overall >= 70:
        return "Silver", "#c0c0c0"
    return "Bronze", "#cd7f32"

# ------------------------------
# Create card as HTML (left) + generate PIL image for download
# ------------------------------
def make_card_html(name, club, club_logo_url, role, overall, stats, color_hex):
    html = f"""
    <div style="
        width: 300px;
        height: 460px;
        margin: auto;
        padding: 15px;
        border-radius: 20px;
        background: linear-gradient(180deg, {color_hex}, #ffffff);
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
        font-family: 'Arial';
        color: black;
        text-align: center;
        border: 4px solid #d4b36a;
    ">
        <!-- Overall & Position -->
        <div style="display:flex; justify-content:space-between;">
            <div style="font-size: 42px; font-weight: 700; margin-left: 5px;">{overall}</div>
            <div style="font-size: 16px; font-weight: 600; margin-top: 10px; margin-right: 5px;">{role}</div>
        </div>
        <!-- Club Logo -->
        <img src="{club_logo_url}" 
             style="width: 70px; height: 70px; margin-top: -10px; object-fit:contain;" />
        <!-- Player Name -->
        <div style="margin-top: 8px; font-size: 22px; font-weight: 700;">{name.upper()}</div>
        <div style="font-size: 14px; margin-top: -4px; opacity: 0.8;">{club}</div>
        <hr style="border-top: 2px solid rgba(0,0,0,0.4); margin: 10px 0;">
        <!-- Stats Grid -->
        <div style="
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            font-size: 18px;
            font-weight: 700;
            margin-top: 10px;
        ">
            <div>PAC</div><div>{stats['pace']}</div>
            <div>SHO</div><div>{stats['shooting']}</div>
            <div>PAS</div><div>{stats['passing']}</div>
            <div>DRI</div><div>{stats['dribbling']}</div>
            <div>DEF</div><div>{stats['defending']}</div>
            <div>PHY</div><div>{stats['physic']}</div>
        </div>
    </div>
    """
    return html

def make_card_image(name, club, club_logo_img, role, overall, stats, color_hex):
    # Render a PIL image (320x480) resembling the card, with club logo and avatar
    W, H = 320, 480
    im = Image.new("RGB", (W, H), color_hex)
    draw = ImageDraw.Draw(im)
    # add a semi-transparent overlay for style
    draw.rectangle([0,0,W,H], fill=color_hex)
    # fonts - use default if custom not found
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
        stat_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        stat_font = ImageFont.load_default()

    # overall
    draw.text((16,16), str(overall), font=title_font, fill="black")
    draw.text((80,20), role, font=small_font, fill="black")
    # club text
    draw.text((80,44), club, font=small_font, fill="black")

    # paste club logo if provided
    if club_logo_img:
        logo = club_logo_img.copy().convert("RGBA")
        logo.thumbnail((70,70), Image.Resampling.LANCZOS)
        im.paste(logo, (16,70), logo)

    # big name
    draw.text((16,150), name, font=stat_font, fill="black")

    # stats grid
    sx = 16; sy = 200; gap = 28
    stat_pairs = [("PAC", stats['pace']), ("SHO", stats['shooting']),
                  ("PAS", stats['passing']), ("DRI", stats['dribbling']),
                  ("DEF", stats['defending']), ("PHY", stats['physic'])]
    for i, (label, val) in enumerate(stat_pairs):
        x = sx + (i%2)*140
        y = sy + (i//2)*gap
        draw.text((x,y), f"{label}: {val}", font=small_font, fill="black")

    return im

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(layout="centered")
st.title("FIFA Player Creator — with Card")

player_name = st.text_input("Enter Character Name", value="PlayerX")

# stats inputs
col1, col2, col3 = st.columns(3)
with col1:
    pace = st.slider("Pace", 1, 99, 70)
    shooting = st.slider("Shooting", 1, 99, 70)
    passing = st.slider("Passing", 1, 99, 70)
with col2:
    dribbling = st.slider("Dribbling", 1, 99, 70)
    defending = st.slider("Defending", 1, 99, 50)
    physic = st.slider("Physical", 1, 99, 60)
with col3:
    gk_diving = st.slider("GK Diving", 1, 99, 2)
    gk_reflexes = st.slider("GK Reflexes", 1, 99, 2)
    gk_handling = st.slider("GK Handling", 1, 99, 2)

movement_acceleration = st.slider("Acceleration", 1, 99, 70)
movement_reactions = st.slider("Reactions", 1, 99, 70)
power_strength = st.slider("Strength", 1, 99, 70)

# optional club selector (or random)
club_choice = st.selectbox("Select Club (or choose Random below)", ["Random"] + clubs)
if club_choice == "Random":
    club = random.choice(clubs)
else:
    club = club_choice

# prepare feature vector
features = np.array([
    pace, shooting, passing, dribbling, defending, physic,
    gk_diving, gk_reflexes, gk_handling,
    movement_acceleration, movement_reactions,
    power_strength
]).reshape(1, -1)
input_features = features

# checkboxes for extra UI
show_potential = st.checkbox("Show predicted potential (regression)", value=True)

if st.button("Generate Card"):
    if player_name.strip() == "":
        st.error("Please enter a name.")
    else:
        # predictions
        role_label = int(role_model.predict(input_features)[0])
        overall_pred = int(round(overall_model.predict(input_features)[0]))
        role_name = role_map.get(role_label, "Unknown")
        potential_val = None
        # if you have a potential regressor, you can predict it here. For now, we use overall+random small bump
        if show_potential:
            potential_val = min(99, overall_pred + random.randint(0,5))

        # rarity & color
        rarity, color_hex = compute_rarity(overall_pred)
        club_logo_url = club_logo_map.get(club, "")
        club_logo_img = fetch_image(club_logo_url, size=(100,100)) if club_logo_url else None

        stats = {
            'pace': pace, 'shooting': shooting, 'passing': passing,
            'dribbling': dribbling, 'defending': defending, 'physic': physic
        }

        

        # show summary
        st.markdown("###  Player Summary")
        st.write(f"**Name:** {player_name}")
        st.write(f"**Club:** {club}")
        st.write(f"**Role:** {role_name}")
        st.write(f"**Overall (pred):** {overall_pred}  |  **Rarity:** {rarity}")
        if potential_val:
            st.write(f"**Potential (est):** {potential_val}")

        # render card HTML
        card_html = make_card_html(player_name, club, club_logo_url, role_name, overall_pred, stats, color_hex)
        card_html = textwrap.dedent(card_html)
        st.components.v1.html(card_html, height=540, scrolling=False)

        # Create PIL image of card and provide download
        card_image = make_card_image(player_name, club, club_logo_img, role_name, overall_pred, stats, color_hex)
        buf = BytesIO()
        card_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

st.markdown("---")

