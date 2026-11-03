# LOCAL-02.4 — Kiểm toán hợp đồng đầu ra và chiều đặc trưng cổng MoE

## 1. Tóm tắt

Phân tích tĩnh xác nhận cả bốn expert fine-tune học với nhãn chuẩn hóa y=(score_raw-5)/5. Miền toán học của đầu ra không đồng nhất: Expert 2 dùng cosine trên vector L2-normalized; Expert 4 có Tanh; Expert 3 kết thúc bằng Linear không giới hạn; Expert 1 dùng tích vô hướng nhưng mã cục bộ không xác nhận embedding được chuẩn hóa. Cùng target không đồng nghĩa cùng miền dự đoán.

| Expert | Score hiện tại | Feature hiện tại | Điều chỉnh cho [B,1] + [B,D_i] |
|---|---|---|---|
| 1 — LAION-CLAP | [B], dot audio/text | Encoder có hai embedding, wrapper không trả | Trích hai embedding trong cùng lần gọi, ghép [audio,text], thêm chiều score |
| 2 — MGA-CLAP | [B,1], cosine | [B,2048] với config đang chọn | Đã tương thích; giữ thứ tự và giá trị |
| 3 — M2D-CLAP | [B,1], Linear không giới hạn | [B,5D_M2D] cross-feature trước head | Đã tương thích về rank; giữ config và thứ tự |
| 4 — BEATs/CLAP/SeqCoAttn | [B], Tanh | [B,1024] pooled cross-attention | Thêm chiều score; giữ feature |

Tổng chiều gating theo các feature này: D_gate = 2*D_CLAP + 2048 + 5*D_M2D + 1024 = 3072 + 2*D_CLAP + 5*D_M2D. D_CLAP và D_M2D chưa được xác minh từ artifact hoặc runtime. Giá trị 7168 trong MoE cũ không thể xác nhận cho các expert fine-tune hiện tại. Không tải checkpoint, không chạy inference hay parity trong task này.

## 2. Expert 1 — LAION-CLAP

### Đầu ra và score

CLAPAudioTextEncoder.forward lấy audio_embedding và text_embedding từ LAION-CLAP API rồi trả cả hai. Expert1.forward tính (audio_features * text_features).sum(dim=-1), nhưng chỉ trả score hạng 1 [B]. Không thấy temperature, clipping hay rescale trong wrapper. (expert1_finetune/Train_Expert1.py, L98–137.)

Đây là dot product, không thể gọi là cosine nếu chưa xác minh L2 normalization. Mã wrapper không chuẩn hóa tường minh. Nếu hai embedding là unit vectors thì score thuộc [-1,1]; nếu không, source không chứng minh có cận hữu hạn. Normalization, width, dtype và device do API trả về cần xác minh runtime: UNVERIFIED.

### Feature đề xuất

Lấy torch.cat([audio_embedding, text_embedding], dim=-1) từ cùng kết quả encoder, audio trước text sau. Nếu mỗi embedding có width D_CLAP thì feature [B,2*D_CLAP]. Embedding được tính cùng lần với score nên không cần chạy encoder hai lần, nhưng Expert1.forward hiện không expose chúng. Adapter phải gọi encoder một lần, tính lại đúng dot product gốc và trả score.unsqueeze(-1), không thêm chuẩn hóa.

| Thuộc tính | Kết luận |
|---|---|
| Score gốc | Dot product, [B] |
| Feature | Embedding audio rồi text, [B,2*D_CLAP] |
| Feature có sẵn trong wrapper? | Không; có sẵn từ CLAPAudioTextEncoder |
| Điều chỉnh | Trích feature cùng forward; reshape score |
| Rủi ro parity | Gọi encoder hai lần hoặc đổi preprocessing/tokenizer/normalization |

Legacy model/Basic_comp.py L244–257 trả score [B,1] và embedding ghép. Biểu thức feature có vẻ tương ứng nếu API/embedding giống nhau, nhưng implementation/checkpoint khác; số học tương đương UNVERIFIED. Phân loại: FEATURE_EXTRACTION_REQUIRED và SHAPE_ADAPTATION_ONLY.

Bằng chứng: Finding: dot product và chỉ trả score. Source: expert1_finetune/Train_Expert1.py; Symbol: CLAPAudioTextEncoder.forward, Expert1.forward; lines 98–137; status STATIC_VERIFIED. Finding: normalization và embedding width của API không được khai báo trong wrapper; cùng source L104–121; status UNVERIFIED.

## 3. Expert 2 — MGA-CLAP

MGAEncoder.forward lấy word-level BERT và frame-level HTSAT từ ASE, đưa từng modality qua MSC/codebook, rồi L2-normalize theo chiều cuối (expert2_finetune/expert2_model.py L17–23). Feature cuối là vector clip-level sau MSC, không phải chuỗi word/frame. Expert2.forward tính dot product với keepdim=True rồi trả score và concat([audio,text]) (L68–71). Vì hai vector đã L2-normalize, score là cosine thuộc [-1,1], không có temperature/clipping trong wrapper.

Cấu hình đang chọn đặt embed_size=1024 (expert2_finetune/MGA-CLAP/settings/inference_example.yaml L1–14); ASE dùng projection/MSC theo proj_size (MGA-CLAP/models/ase_model.py L103–144). Feature là [B,2048] trong config đó: audio MSC 1024 trước text MSC 1024. Width có thể đổi với config khác; lấy từ config/model dựng, không coi 2048 là hằng số toàn cục. Wrapper không ép dtype hoặc device; score/feature theo dtype/device đầu ra ASE và vị trí module. Giá trị runtime cần kiểm tra: UNVERIFIED.

| Thuộc tính | Kết luận |
|---|---|
| Score | Cosine [B,1] |
| Feature | [audio_MSC, text_MSC], [B,2048] với config hiện tại |
| Chuẩn hóa | Mỗi nửa chuẩn hóa riêng; concat không chuẩn hóa lại |
| Feature có sẵn? | Có |
| Điều chỉnh | Không đổi rank/width/thứ tự với config hiện tại |
| Rủi ro parity | Nhầm vector MSC cuối với chuỗi HTSAT/BERT; cấp sai waveform hoặc tokenizer |

Legacy model/Basic_comp.py L261–272 dùng cùng mô thức MSC → normalize → dot → concat audio/text. Tương ứng ở mức biểu thức source, nhưng chưa chứng minh checkpoint hoặc output parity: STATIC_VERIFIED về công thức, UNVERIFIED về số học thực tế.

Bằng chứng: expert2_finetune/expert2_model.py, MGAEncoder.forward L17–23 và Expert2.forward L68–71; STATIC_VERIFIED. Cấu hình width: expert2_finetune/MGA-CLAP/settings/inference_example.yaml L1–14; STATIC_VERIFIED cho cấu hình đang chọn. Chiều ASE/MSC: expert2_finetune/MGA-CLAP/models/ase_model.py L103–144; STATIC_VERIFIED.

## 4. Expert 3 — M2D-CLAP

### Cross-feature

encode_modalities chuẩn hóa text_emb và audio_emb bằng F.normalize(..., dim=-1), rồi trả thứ tự (text_emb,audio_emb) (expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py L210–242). forward gán hai tensor cho a_f/t_f tùy concat_assignment, tạo cross-feature, chạy alignment head FP32, rồi trả (predicted_score,cross_features) (L244–266).

Với a_f,t_f mỗi tensor [B,D], các block là:
- cosine = sum(a_f*t_f) có shape [B,1];
- abs-diff = abs(a_f-t_f), [B,D];
- product = a_f*t_f, [B,D];
- cosine được expand/lặp D lần, [B,D];
- concat theo [a_f, t_f, abs-diff, product, repeated-cos], tổng [B,5D].

Cross-feature chính là input alignment head và cũng là feature trả ra; không normalize sau concat. Forward không chuyển tensor sang device khác; score và feature nằm trên device của embedding/model khi đầu vào tương thích. Embedding/cross-feature được ép FP32 và head chạy autocast disabled, nên score/feature dự kiến FP32 (moescore_expert3.py L242, L261–266).

| concat_assignment | a_f | t_f | Hai block đầu trong feature |
|---|---|---|---|
| released (giá trị mặc định trong constructor) | text_emb | audio_emb | text, audio |
| semantic | audio_emb | text_emb | audio, text |

diff, product và cosine đối xứng theo đổi audio/text; hai block đầu thì không. Linear head nhạy theo vị trí, vì vậy không đổi thứ tự với checkpoint đã fine-tune. Constructor chấp nhận released/semantic và lưu lựa chọn trong model_config (moescore_expert3.py L98–105, L123–128, L158–166). Cấu hình fine-tuning được kiểm tra đặt concat_assignment là released và train entry point truyền cfg["concat_assignment"] vào MoescoreExpert3 (expert3_finetune/XACLE_2027_baselinemodel/configs/expert3_moescore.json L40; train_expert3.py L386–390). Checkpoint hiện hành cần giữ đúng thứ tự released.

### Dimension và score

D được suy từ output của audio_proj và text_proj, hoặc từ dimension config của text encoder nếu checkpoint không có text_proj. Constructor kiểm tra hai chiều bằng nhau, đặt embedding_dim=D và cross_feature_dim=5D (L132–147). Cả hai thuộc tính đọc được sau khởi tạo, không cần forward. D thay đổi theo checkpoint/model family; giá trị số chưa được artifact-verified.

Alignment head là Linear(5D,512) → ReLU → Dropout → Linear(512,128) → ReLU → Dropout → Linear(128,1) (L148–156). Không có Tanh/Sigmoid/clipping sau Linear cuối, nên score [B,1] không bị chặn toán học. Dropout head ảnh hưởng score khi train mode, không ảnh hưởng cross_features trả ra. Cross-feature là FP32 do embeddings được cast float và head chạy khi autocast disabled (L242, L261–266).

Legacy Basic_comp Expert3 có head input cố định 3072 nhưng dùng encoder/API cũ; con số này không chứng minh D hay feature width của mô hình hiện tại. Phân loại ARCHITECTURAL_DIFFERENCE; parity UNVERIFIED.

Bằng chứng: dimension/head moescore_expert3.py L132–166; normalize/return order L210–242; feature ordering/score/output L244–266. Kiến trúc STATIC_VERIFIED, chiều thực tế của checkpoint và parity UNVERIFIED.

## 5. Expert 4 — BEATs + CLAP + SeqCoAttn

Audio/text encoder tạo sequence width 512; SeqCoAttn có hai MultiheadAttention(dim=512,nhead=8,batch_first=True), một audio-query/text-key-value và một text-query/audio-key-value. Hai nhánh dùng sequence đầu vào ban đầu, không truyền output nhánh thứ nhất vào nhánh thứ hai. Mỗi output được AdaptiveMaxPool1d(1); concat pooled audio-query trước text-query sau thành [B,1024] (expert4_finetune/seq_coattn.py L6–22). Expert4.forward trả (head(fused),fused) (expert4_finetune/model.py L20–24).

Feature fusion không được L2-normalize. Padding masks Boolean được dùng làm key_padding_mask; True là vị trí padding theo API MultiheadAttention, khớp mask do encoder tạo (seq_coattn.py L17–21; text_encoder.py L111–124; audio_encoder.py L42–76). Mask chặn key/value padding, nhưng không lọc query outputs trước max pooling; query ở vị trí padding vẫn có thể tham gia pooling. Đây là rủi ro parity cần kiểm tra, không sửa trong audit.

Regression head: Linear(1024,2048) → ReLU → Dropout(0.3) → Linear(2048,1) → Tanh → squeeze(-1), nên score [B] và miền toán học [-1,1] (seq_coattn.py L25–31). squeeze chỉ bỏ chiều cuối singleton, batch B=1 vẫn còn [1].

| Thuộc tính | Kết luận |
|---|---|
| Feature | Pooled audio-query-to-text rồi pooled text-query-to-audio |
| Width | 512+512=1024 |
| Chuẩn hóa | Không L2 normalize fusion output |
| Score | Tanh regression, [B] |
| Điều chỉnh | score.unsqueeze(-1); giữ nguyên feature |
| Batch 1 | Trục batch còn nguyên sau squeeze(-1) |

Trainer dùng CUDA autocast FP16 trong training nhưng ép predictions.float() trước loss; inference kiểm tra score [B], fused [B,1024] (trainer.py L275–289; inference.py L14–25). Forward không chuyển output sang device khác; score/features theo thiết bị của model/input. Dtype feature khi autocast có thể khác; ngoài autocast thường theo dtype model. Parity phải so trong eval cùng precision; dtype/device thực tế cần kiểm tra runtime.

Bằng chứng: attention/mask/pooling expert4_finetune/seq_coattn.py L6–22; Tanh/rank L25–31; output Expert4 expert4_finetune/model.py L20–24; training/inference trainer.py L275–289, inference.py L14–25. STATIC_VERIFIED.

## 6. So sánh score giữa experts

| Expert | Score gốc | Miền toán học | Target train | Validation/inference |
|---|---|---|---|---|
| 1 | Dot audio/text [B] | Cận không xác lập; [-1,1] chỉ nếu unit vectors | (raw-5)/5 | Validation inverse pred*5+5 (Train_Expert1.py L247–275) |
| 2 | Dot hai vector normalized [B,1] | [-1,1] | (raw-5)/5 (expert2_data.py L68–83) | Validation inverse *5+5 (Train_Expert2.py L81–93) |
| 3 | Linear head [B,1] | Không giới hạn | (raw-5)/5 (moescore_expert3_dataset.py L98–115) | Inverse *5+5 (train_expert3.py L309–321; inference_expert3.py L61–67) |
| 4 | Regression + Tanh [B] | [-1,1] | (raw-5)/5 (expert4_finetune/dataset.py L107–115) | Validation inverse *5+5 (trainer.py L80–98) |

Target của cả bốn theo source là chuẩn hóa 0–10 thành [-1,1]. Nhưng score Expert 1 chưa xác nhận bounded và Expert 3 không bounded. Inverse transform dùng cho metric thang gốc không phải bằng chứng adapter phải chuyển score trước MoE. Prediction range thực nghiệm, phân phối và hiệu chuẩn đều UNVERIFIED vì chưa có checkpoint/output.

## 7. So sánh feature giữa experts

| Expert | Nguồn | Width | Chuẩn hóa | Ý nghĩa |
|---|---|---:|---|---|
| 1 | LAION-CLAP audio embedding nối text embedding | 2*D_CLAP | API chưa xác minh; wrapper không normalize | Cặp vector global của cùng audio-caption |
| 2 | MSC audio nối MSC text | 2048 trong config hiện tại | Mỗi nửa L2-normalized | Hai embedding global sau codebook |
| 3 | audio/text theo concat_assignment, abs-diff, product, cosine lặp | 5*D_M2D | Hai embedding đầu vào normalize; concat không normalize | Cross-modal interaction trước alignment head |
| 4 | Pooled hai chiều cross-attention | 1024 | Không normalize | Vector fusion theo attention giữa chuỗi audio-text |

Feature khác nhau về ý nghĩa, phân phối và scale. Expert 2 có từng nửa norm 1; concat toàn vector có norm gần sqrt(2). Expert 3 chứa cosine lặp D lần và product; norm có thể phụ thuộc D. Expert 4 không normalize tường minh. Expert 1 normalization chưa xác minh. Gate có LayerNorm sau Linear 512 chiều; điều này không chứng minh raw feature scales tương đương (model/MOE_model.py L11–24). Không thêm projection hay normalization trong task này.

## 8. Legacy và fine-tuned output tương thích

| Expert | Legacy output | Fine-tuned output | Phân loại và ảnh hưởng |
|---|---|---|---|
| 1 | Dot [B,1], concat embedding | Dot [B], encoder không expose feature | FEATURE_EXTRACTION_REQUIRED + SHAPE_ADAPTATION_ONLY; numerical parity UNVERIFIED |
| 2 | Dot [B,1], concat audio/text normalized MSC | Cùng dạng/thứ tự; [B,1]+[B,2048] theo config | EQUIVALENT ở công thức; API/input keys khác, runtime parity UNVERIFIED |
| 3 | M2D API cũ, head input cố định 3072, score [B,1] | MoescoreExpert3 hiện tại, dynamic 5D, assignment cấu hình được | ARCHITECTURAL_DIFFERENCE; không giả định checkpoint tương thích |
| 4 | TemporalAlignModel + Projection config-driven, score squeeze [B] | BEATs/CLAP + SeqCoAttn, head có Tanh, score [B] | ARCHITECTURAL_DIFFERENCE; width bằng nhau không chứng minh hidden path tương đương |

Nguồn legacy: model/Basic_comp.py L13–42, L46–83, L86–128, L165–198, L202–257, L261–324, L328–336. Expert3 legacy gọi encode_clap_audio_multiscale (L191–198), trong khi current model gọi encode_clap_audio (moescore_expert3.py L222–229); legacy head cố định in_dim=3072 (Basic_comp.py L282–291), nên không có bằng chứng tương thích với checkpoint hiện tại. Expert4 legacy head dùng cấu hình và chỉ bật Tanh nếu range_clipping được bật (L46–83), khác head fine-tuned có Tanh cố định. Legacy SeqCoAttn khai báo LayerNorm/residual projections nhưng active residual/norm calls đang comment (Basic_comp.py L86–128); output width bằng nhau không xác nhận cùng feature/checkpoint semantics.

Generic loader/call path trong MOESCORE không thay thế API dựng và khôi phục từng implementation fine-tuned. Adapter tương lai cần dùng implementation đúng rồi chuẩn hóa giao diện; audit này không sửa legacy code.

## 9. Kiểm toán gating hiện hành

GatingNetwork là Linear(input_feature_dim,512) → LayerNorm(512) → ReLU → Dropout(0.1) → Linear(512,num_experts) → Softmax(dim=-1) (model/MOE_model.py L11–24). MOEModel khởi tạo gate với 4 experts dù constructor mặc định 3 (L60–67). Forward nối feature theo thứ tự E1,E2,E3,E4; gate chỉ nhận feature, không nhận score; weights có shape [B,4], softmax trên trục expert. Kiến trúc cho phép trọng số phụ thuộc từng sample nhưng chưa chứng minh gate đã học được tính biến thiên có ích.

Công thức dự định: final = Σ score_i * weights[:,i:i+1] (L71–89). Nếu mọi score là [B,1], kết quả [B,1]. Expert4 fine-tuned trả [B], nên score4 [B] * weights[:,3:4] [B,1] broadcast thành [B,B] với B>1; tổng sẽ có rank/semantics sai. Expert4 legacy cũng squeeze(-1) về [B] (model/Basic_comp.py L74–83), nên đây là lỗi tiềm tàng của phép gộp hiện tại, không chỉ do tích hợp fine-tuned version. B=1 che lỗi. Adapter/giao diện tương lai phải chuẩn hóa score về [B,1] trước khi gộp.

Experts được đặt eval và requires_grad=False tại constructor, expert forward ở trong no_grad (L55–58, L71–76). Gating call nằm ngoài no_grad nên gate vẫn nhận gradient (L78–87). Tuy nhiên Train_MOE gọi model.train() mỗi epoch (Train_MOE.py L108–110); PyTorch đệ quy chuyển con về train mode, trừ module override. MOEModel không override train. Dropout có thể bật lại dù experts eval lúc khởi tạo; cần duy trì eval mode riêng khi triển khai gating training.

Gate dùng input-dependent features, không dùng score làm feature. Softmax từng hàng đảm bảo trọng số không âm và tổng mỗi hàng bằng 1 về mặt toán học. Train_MOE.py L21–26 định nghĩa loss là 1 trừ normalized entropy và cộng loss này vào objective (L127–132); tối thiểu hóa nó khuyến khích entropy cao/uniform hơn, nhưng không chứng minh weights có ý nghĩa phụ thuộc sample. Cần kiểm tra sau training rằng gate chưa sụp thành trọng số hằng.

Bằng chứng: model/MOE_model.py L11–24, L55–89; Train_MOE.py L108–110, L127–132. Cấu trúc STATIC_VERIFIED; hành vi gate đã học UNVERIFIED.

## 10. Suy ra chiều input gate

Theo feature đề xuất:
- D1 = 2*D_CLAP.
- D2 = 2*1024 = 2048 trong cấu hình MGA hiện tại.
- D3 = 5*D_M2D.
- D4 = 2*512 = 1024.
- D_gate = D1+D2+D3+D4 = 3072 + 2*D_CLAP + 5*D_M2D.

D2 đến từ config hiện tại; D4 từ hai nhánh attention width 512; D1 và D3 phụ thuộc model/checkpoint. D3 đọc được từ model.cross_feature_dim sau constructor. D1 cần lấy từ model/API LAION thực tế; chưa xác nhận attribute cụ thể trong fine-tuned implementation. Không thể tính chính xác một số D_gate tĩnh trước khi dựng/đọc hai model đó.

MOE_model.py L60 gán cứng 7168 mà không suy từ expert instances. Phân tích số 1024+2048+3072+1024 có thể là giả thuyết về width từng legacy expert, nhưng source không khẳng định cách tính này và 3072 của legacy Expert3 là input head, không chứng minh output feature width của mô hình hiện tại. Vì vậy 7168 không được xác nhận, có rủi ro mismatch với Linear input. Tương lai nên lấy D_i từ adapter/model/config sau dựng, tính tổng, rồi assert từng feature width và concat width trước train.

Raw concat giữ thông tin input-dependent và không thêm tầng học được, nhưng feature scales khác nhau như mục 7. LayerNorm sau projection không đảm bảo cân bằng feature-wise. Bất kỳ projection/normalization mới nào là quyết định kiến trúc riêng cần benchmark.

Bằng chứng: gate hardcode/cat model/MOE_model.py L60–80; Expert3 dynamic dim moescore_expert3.py L132–147; MGA config inference_example.yaml L1–14; attention width expert4_finetune/seq_coattn.py L6–22. Công thức D_gate DERIVED; D_CLAP, D_M2D và tính đúng của 7168 UNVERIFIED.

## 11. Hợp đồng output MoE đề xuất

Mỗi adapter trả tuple có thứ tự cố định (score, features):
- score là Tensor [B,1], một dự đoán gốc cho mỗi audio-caption; không calibration, clipping hay rescale.
- features là Tensor [B,D_i], feature sample-level theo định nghĩa expert; không đổi ordering hoặc normalization gốc.
- Cùng batch size, cùng thứ tự sample, device/dtype tương thích.
- Width feature phải khớp D_i đã khai báo/đọc từ model; mismatch cần fail-fast.
- Score/features hữu hạn và cùng sample identity; hiện chưa có validator chung.
- Expert1 và Expert4 thêm chiều score cuối; không đổi giá trị.
- Expert2 và Expert3 giữ score; tất cả experts giữ feature.
- Concatenate features theo E1,E2,E3,E4; tạo weights [B,4]; xếp scores có hình dạng thống nhất; weighted sum trả [B,1].

Đây là contract đề xuất, chưa implement hoặc test.

## 12. Chiến lược trích feature cho adapters

| Expert | Feature đề xuất | Nguồn/ý nghĩa | Width | Có sẵn? / effort |
|---|---|---|---:|---|
| 1 | concat(audio_embedding,text_embedding) | CLAPAudioTextEncoder; cặp embedding dùng tính dot | 2*D_CLAP | Có trong encoder, không trong wrapper; adapter lấy cùng một lần gọi |
| 2 | concat(audio_MSC,text_MSC) | Expert2.forward; normalized global vectors | 2048 hiện tại | Trả trực tiếp; effort tối thiểu |
| 3 | cross_features nguyên bản | MoescoreExpert3.forward; pair + interaction blocks | model.cross_feature_dim=5D | Trả trực tiếp; giữ assignment/order |
| 4 | fused nguyên bản | Expert4.forward; pooled bidirectional cross-attention | 1024 | Trả trực tiếp; chỉ đổi rank score |

Với Expert1, adapter khái niệm nên dùng một lần gọi encoder để tính đồng thời dot gốc và feature concat, không gọi encoder lần hai. Có thể composition/wrapper mà không đổi kiến trúc; lựa chọn API thuộc LOCAL-02.5. Không thêm projection hoặc normalization.

## 13. Hành vi expert khi đóng băng

Ba cơ chế độc lập:
1. requires_grad=False ngăn cập nhật tham số.
2. eval() đổi hành vi module như dropout.
3. no_grad() không lưu graph cho forward expert.

MOE hiện bật freeze/eval lúc khởi tạo và no_grad quanh expert; gate chạy ngoài block nên vẫn nhận gradient từ weighted loss. Feature không cần gradient về expert. Parent model.train() có thể bật lại train mode theo cây module. Tương lai cần giữ experts eval kể cả khi MOE train, ví dụ bằng setter/override ở tầng MoE; không sửa trong task này. Không dùng inference_mode cho feature đi vào gate trainable nếu chưa kiểm interaction autograd; no_grad phù hợp với cách hiện tại.

Expert2 có override train riêng để quản lý mode encoder/layer (expert2_model.py L50–66), không giải quyết mode các expert khác. requires_grad=False không tự chuyển model sang eval.

## 14. An toàn shape và số học

- Expert1/4 [B] cần nâng thành [B,1]; nếu không Expert4 tạo broadcasting [B,B].
- Expert4 squeeze(-1) vẫn giữ batch khi B=1; cần test output adapter/gate ở B=1 và B>1.
- Batch ordering phải giống nhau ở audio, text và labels; cùng B không phát hiện lệch identity.
- Expert3 ép embedding/cross feature float32; Expert4 training có autocast nên dtype có thể thay đổi. Adapter/gate cần dtype và device tương thích khi cat/stack.
- MOE forward không có check rõ cho empty batch, NaN/Inf, score rank hay feature width.
- Expert3 không bounded và Expert1 chưa rõ range. Chỉ khi toàn score bounded và weights nonnegative cộng 1 thì mixture đảm bảo convex range; điều kiện này chưa được chứng minh cho cả bốn.
- Expert4 mask chỉ áp dụng key/value; query padding có thể vào max pool. Cần test độ nhạy theo padding/length.

Bằng chứng rank/broadcast: model/MOE_model.py L82–87 và expert4_finetune/seq_coattn.py L25–31. Không thấy guard trong forward MOE_model.py L71–89; STATIC_VERIFIED cho đoạn mã đã kiểm tra.

## 15. Kế hoạch kiểm chứng runtime sau này

Không chạy các kiểm tra này trong LOCAL-02.4.

| Kiểm tra | Cần checkpoint fine-tuned? | Yêu cầu |
|---|---|---|
| Đọc model attributes/config và dimension | Có thể không nếu constructor chỉ dùng pretrained resource | Pretrained/config/dependencies |
| Return pair, rank/dtype/device/finite, B=1 và B>1 | Có | Fine-tuned checkpoint và sample cố định |
| Score parity standalone vs adapter | Có | Cùng preprocessing, eval, precision |
| Feature parity và E3 ordering/assignment | Có | Checkpoint và intermediate comparison |
| E4 mask/length behavior, repeated deterministic output | Có | Checkpoint và batch lengths khác nhau |
| Gate concat width/order, score stack, weights [B,4], row sum, final [B,1] | Cấu trúc có thể test bằng tensors giả; full test cần adapters | Contract implementation |
| Gate input-dependence/collapse | Collapse cần gate đã train | Nhiều samples và trained gate |
| Freeze: params bất biến, gate grad khác 0 | Không nhất thiết cần fine-tuned weights để test cấu trúc | Adapters và optimizer gating-only |
| NaN/Inf, range, feature stats, repeated eval parity | Cần restored experts và inputs thực | Representative batches |

Parity cần cùng audio-caption/preprocessing, eval, không augmentation; đặt tolerance trước. Expert1 cần xác minh normalization/width; Expert3 xác nhận concat_assignment; Expert4 xác nhận padding mask. Khác biệt weights của gate chưa train không chứng minh gate học tốt.

## 16. Rủi ro và câu hỏi mở

1. D_CLAP và normalization LAION: runtime API/checkpoint quyết định width và score bound.
2. D_M2D và concat_assignment: phải khớp chính xác checkpoint fine-tune.
3. Gate width 7168: chưa gắn với widths adapters.
4. Expert4 score rank cần chuẩn hóa để tránh [B,B].
5. model.train() của parent có thể bật lại dropout expert.
6. Expert4 key-padding mask không loại query padding trước pooling; parity cần test.
7. Target đều chuẩn hóa nhưng score Expert3 unbounded, Expert1 chưa rõ.
8. Feature scales/semantics dị thể; internal LayerNorm không chứng minh chúng cân bằng.
9. Checkpoint/API parity chỉ xác nhận bằng standalone-adapter output comparison.
10. Không có fail-fast chung cho shape, width, finite values hoặc sample identity.

## 17. Khuyến nghị cho LOCAL-02.5

- Dùng đúng implementation/checkpoint factory và input mapping riêng từng expert theo LOCAL-02.3; không tái dùng token tensor chỉ vì cùng tên.
- Thiết kế forward adapter lấy score gốc và feature nêu trên; Expert1 trích embeddings trong cùng lần encoder call.
- Cố định thứ tự experts E1,E2,E3,E4; chuẩn hóa rank score thành [B,1], kiểm batch/width/device/dtype/finite trước gate.
- Đọc D2 từ config, D3 từ cross_feature_dim, D1 từ model/API LAION, D4 từ fusion width; tính gate input từ tổng, không hardcode 7168.
- Bắt đầu bằng raw feature concat; mọi normalization/projection/calibration cần thí nghiệm riêng.
- Giữ experts eval cả khi parent MOE chuyển train; freeze params, dùng no_grad cho expert forward, giữ autograd cho gate.
- Không tuyên bố checkpoint parity trước khi so score/feature standalone với adapter.

Không tạo adapter hoặc sửa production code trong task này.

## 18. Tham chiếu nguồn

Line references trỏ tới working tree được kiểm tra trong LOCAL-02.4.

| File | Symbol/đoạn | Lines | Bằng chứng |
|---|---|---:|---|
| expert1_finetune/Train_Expert1.py | CLAPAudioTextEncoder; Expert1; train/validation | 98–137, 247–275 | Embeddings, dot, return rank, target |
| model/Basic_comp.py | TemporalAlignModel, Projection, SeqCoAttn, experts | 13–128, 165–198, 202–257, 261–324, 328–336 | Legacy output paths |
| expert2_finetune/expert2_model.py | MGAEncoder; Expert2; train override | 17–23, 50–71 | MSC, normalize, output, modes |
| expert2_finetune/MGA-CLAP/settings/inference_example.yaml | active config | 1–14 | embed_size |
| expert2_finetune/MGA-CLAP/models/ase_model.py | ASE projection/MSC | 103–144 | Dimension flow |
| expert2_finetune/expert2_data.py | target transform | 68–83 | Normalized target |
| expert2_finetune/Train_Expert2.py | train/validation | 81–93, 180–205 | Score transforms |
| expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py | constructor/head/model_config | 98–166 | D, 5D, assignment |
| expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py | encode_modalities/forward | 210–266 | Normalize, concat, returns |
| expert3_finetune/XACLE_2027_baselinemodel/configs/expert3_moescore.json | concat_assignment | 40 | Fine-tuning config chọn released |
| expert3_finetune/XACLE_2027_baselinemodel/train_expert3.py | MoescoreExpert3 construction | 386–390 | Truyền config vào model |
| expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py | target | 98–115 | Normalized target |
| expert3_finetune/XACLE_2027_baselinemodel/train_expert3.py | loss/validation | 245–254, 309–321 | Training/inverse transform |
| expert3_finetune/XACLE_2027_baselinemodel/inference_expert3.py | inference | 61–67 | Raw-scale transform |
| expert4_finetune/seq_coattn.py | SeqCoAttn/RegressionHead | 6–31 | Width, heads, masks, pooling, Tanh |
| expert4_finetune/model.py | Expert4.forward | 20–24 | Return pair |
| expert4_finetune/audio_encoder.py | AudioEncoder | 42–76 | Audio width/mask |
| expert4_finetune/text_encoder.py | TextEncoder | 111–124 | Text width/mask |
| expert4_finetune/dataset.py | target | 107–115 | Normalized target |
| expert4_finetune/trainer.py | train/validation | 80–98, 275–289 | Metric/autocast |
| expert4_finetune/inference.py | inference checks | 14–25 | Rank and width |
| model/MOE_model.py | gate/MOEModel.forward | 11–24, 55–89 | Gate, width, freeze, aggregation |
| Train_MOE.py | train loop/loss | 108–110, 127–132 | Parent mode, optimize gate |

### Trạng thái xác minh

- Output paths, công thức, ranks, widths cố định/động và gate: STATIC_VERIFIED theo source references.
- D_CLAP, D_M2D số thực tế, LAION embedding normalization, empirical score ranges, checkpoint restoration, adapter parity và gate đã học: UNVERIFIED.
- Không có khẳng định ARTIFACT_VERIFIED hoặc RUNTIME_VERIFIED.
- Không sửa production code, architecture, checkpoint hoặc training algorithm; chỉ tạo báo cáo này.
- Trạng thái cuối: **PASS — STATIC AUDIT COMPLETE**. Trạng thái này xác nhận source audit, không chứng minh checkpoint loadability hoặc numerical inference parity.


