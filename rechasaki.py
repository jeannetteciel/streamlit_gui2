import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, KNNBasic, dump

# Đọc dữ liệu
df_product = pd.read_csv("San_pham.csv")
df_review = pd.read_csv("Danh_gia.csv")
df_customer = pd.read_csv("Khach_hang.csv")
df_collab = pd.read_csv("collab_df.csv")

# Chuẩn bị dữ liệu cho Content-Based Filtering
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_product['mo_ta'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Hàm đề xuất dựa trên cosine similarity
def get_recommendations(sp_id, cosine_sim, nums=5):
    idx = df_product.index[df_product['ma_san_pham'] == sp_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums+1]
    sp_indices = [i[0] for i in sim_scores]
    return df_product.iloc[sp_indices]

# Chuẩn bị dữ liệu cho Collaborative Filtering
df_collab = df_collab.rename(columns={"ma_khach_hang": "user_id", "ma_san_pham": "item_id", "so_sao": "rating"})
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(df_collab[['user_id', 'item_id', 'rating']], reader)

# Huấn luyện mô hình KNN
algorithm = KNNBasic()
trainset = data.build_full_trainset()
algorithm.fit(trainset)

# Lưu mô hình vào file
dump.dump('knn_model.pkl', algo=algorithm)

# Hàm đề xuất sản phẩm dựa trên Collaborative Filtering
def recommend_products(user_id, top_n=5, threshold=3.5):
    if user_id not in df_collab['user_id'].unique():
        return pd.DataFrame(columns=['ma_san_pham', 'ten_san_pham', 'EstimateScore'])

    all_items = df_collab['item_id'].unique()
    rated_items = df_collab[df_collab['user_id'] == user_id]['item_id'].unique()
    items_to_predict = list(set(all_items) - set(rated_items))

    predictions = []
    for item_id in items_to_predict:
        try:
            est_score = algorithm.predict(user_id, item_id).est
            if est_score >= threshold:
                predictions.append((item_id, est_score))
        except Exception as e:
            print(f"Error predicting for user {user_id} and item {item_id}: {e}")
            pass

    if not predictions:
        return pd.DataFrame(columns=['ma_san_pham', 'ten_san_pham', 'EstimateScore'])

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    recommended_items = pd.DataFrame(predictions, columns=['ma_san_pham', 'EstimateScore'])
    recommended_items = recommended_items.merge(df_product, on='ma_san_pham', how='left')

    return recommended_items[['ma_san_pham', 'ten_san_pham', 'gia_ban', 'gia_goc', 'phan_loai', 'mo_ta', 'diem_trung_binh', 'EstimateScore']]

# Giao diện chính của Streamlit
st.set_page_config(page_title="Hệ thống Recommender HASAKI.VN", layout="wide")

# Hiển thị banner
st.image("hasakibanner.png", use_column_width=True)

st.title("Hệ thống Recommender sản phẩm mỹ phẩm HASAKI.VN")
st.markdown("""
**HASAKI.VN** là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với hệ thống cửa hàng trải dài trên toàn quốc.  
Khách hàng có thể lựa chọn sản phẩm, xem đánh giá/nhận xét và đặt mua sản phẩm.
""")

# Tabs phân chia chức năng
tab1, tab2, tab3 = st.tabs(["Giới thiệu", "Đăng nhập", "Sản phẩm và Đề xuất"])

# Tab 1: Giới thiệu
with tab1:
    st.header("Chào mừng đến với hệ thống Recommender HASAKI.VN")
    st.write("Hệ thống giúp bạn khám phá và tìm kiếm các sản phẩm mỹ phẩm phù hợp nhất với nhu cầu của mình.")

# Tab 2: Đăng nhập
with tab2:
    st.subheader("Đăng nhập")
    customer_code = st.text_input("Nhập mã khách hàng", placeholder="Ví dụ: 3000")
    if customer_code:
        customer_info = df_customer[df_customer['ma_khach_hang'] == int(customer_code)]
        if not customer_info.empty:
            st.success(f"Xin chào {customer_info['ho_ten'].values[0]}!")
            st.write("Thông tin của bạn:")
            st.write(customer_info)
            
            # Đề xuất sản phẩm từ Collaborative Filtering
            st.write("Sản phẩm đề xuất (Dựa trên khách hàng tương tự):")
            recommended_products_collab = recommend_products(int(customer_code))
            if not recommended_products_collab.empty:
                st.dataframe(recommended_products_collab)
            else:
                st.write("Không có sản phẩm phù hợp.")
        else:
            st.warning("Mã khách hàng không hợp lệ! Vui lòng kiểm tra lại.")

# Tab 3: Sản phẩm và Đề xuất
with tab3:
    st.subheader("Chọn sản phẩm")
    # Chọn sản phẩm theo tên
    selected_product = st.selectbox("Chọn sản phẩm", df_product['ten_san_pham'])
    product_info = df_product[df_product['ten_san_pham'] == selected_product]

    st.write("Thông tin sản phẩm đã chọn:")
    st.write(product_info[['ten_san_pham', 'gia_ban', 'mo_ta']])

    # Hiển thị sản phẩm đề xuất (Dựa trên độ tương tự)
    st.write("Sản phẩm đề xuất (Dựa trên độ tương tự):")
    recommended_products = get_recommendations(product_info['ma_san_pham'].values[0], cosine_sim)
    for _, row in recommended_products.iterrows():
        st.markdown(f"**{row['ten_san_pham']}**")
        st.write(f"Giá bán: {row['gia_ban']} | Điểm trung bình: {row['diem_trung_binh']}")
        st.write(f"Mô tả: {row['mo_ta'][:100]}...")  # Hiển thị 100 ký tự đầu của mô tả
        st.divider()

    # Hiển thị đánh giá sản phẩm
    st.write("Đánh giá sản phẩm:")
    product_reviews = df_review[df_review['ma_san_pham'] == product_info['ma_san_pham'].values[0]]
    if not product_reviews.empty:
        product_reviews = product_reviews[['ma_khach_hang', 'noi_dung_binh_luan', 'so_sao']]
        product_reviews = product_reviews.rename(columns={
            'ma_khach_hang': 'Mã Khách Hàng', 
            'noi_dung_binh_luan': 'Nội Dung Bình Luận', 
            'so_sao': 'Số Sao'
        })
        st.dataframe(product_reviews)
    else:
        st.write("Không có đánh giá cho sản phẩm này.")
