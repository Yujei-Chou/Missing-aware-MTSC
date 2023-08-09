# Missing-aware-MTSC
Missing-aware Multivariate Time Series Forecasting CKD Patients' Future Disease Condition
# Preprocessing:
- LAB data:
    ```
    python LAB_Preprocessing.py
    ```
- Patient data:
    ```
    python Patient_Preprocessing.py
    ```
- CKD data:
    ```
    python CKD_Preprocessing.py 
    ```
- Note:
  1. 前兩個前處理有涉及到處理mdb檔案必須先安裝[驅動程式](https://www.microsoft.com/en-us/download/details.aspx?id=54920)
  2. IRB資料放在Nas(186)中的Dataset/CKD_IRB_Data
  3. [前處理後的檔案與模型權重](https://drive.google.com/file/d/1uOpG7RPLJpaBbq8Tnj06ZA5YUfgQPBzc/view?usp=sharing)
  4. [醫師作答問卷結果](https://drive.google.com/drive/folders/1-_cdz3oXnGPPlMc9417JMU6RdArAR4e9?usp=sharing)
# Model:
- **Ours:** 
    - Framework:
        ![](https://hackmd.io/_uploads/rJspdLy23.png)

    - Pretrain:
        - Pretext Task:
            ```
            python main_DL.py --status train --training_type semi_balance --testing_type balance --weight_path weight/pretrain/Ours_1yearDrop30%_pretrain_noFM --pretrain --epochs 100 --if_FM 0
            ```
        - Fusion Module + Pretext Task:
            ```
            python main_DL.py --status train --training_type semi_balance --testing_type balance --weight_path weight/pretrain/Ours_1yearDrop30%_pretrain --pretrain --epochs 100
            ```
    - Finetune:
        - Pretext Task:
            ```
            python main_DL.py --status train --training_type balance --testing_type balance --pretrain_path weight/pretrain/Ours_1yearDrop30%_pretrain_noFM --weight_path weight/finetune/Ours_1yearDrop30%_finetune_noFM --finetune --learning_rate 0.00001 --if_FM 0
            ```
        - Fusion Module + Pretext Task:
            ```
            python main_DL.py --status train --training_type balance --testing_type balance --pretrain_path weight/pretrain/Ours_1yearDrop30%_pretrain --weight_path weight/finetune/Ours_1yearDrop30%_finetune --finetune --learning_rate 0.00001
            ```

    - Not Pretrain:
        - Baseline:
            ```
            python main_DL.py --status train --model MAMTSC --if_FM 0 --training_type balance --testing_type balance --weight_path weight/notpretrain/Transformer_1yearDrop30%
            ```
        - Fusion Module:
            ```
            python main_DL.py --status train --model MAMTSC --training_type balance --testing_type balance --weight_path weight/notpretrain/Ours_1yearDrop30%_onlyFM
            ```
    - Only Testing:
        - Balance:
            ```
            python main_DL.py --status test --testing_type balance --weight_path weight/finetune/Ours_1yearDrop30%_finetune
            ```
        - Imbalance:
            ```
            python main_DL.py --status test --testing_type imbalance --weight_path weight/finetune/Ours_1yearDrop30%_finetune
            ```
        - Custom:
            ```
            python main_DL.py --status test --testing_type custom --weight_path weight/finetune/Ours_1yearDrop30%_finetune --testing_path dataset/preprocessed_data/Experts_Questions/Sample100_allform
            ```
- DL-based:
    - **LSTM:**
        ```
        python main_DL.py --status train --model LSTM --if_FM 0 --training_type balance --testing_type balance --weight_path weight/notpretrain/LSTM_1yearDrop30% --d_model 32 --batch_size 64 --epochs 100
        ```
    - **Transformer:**
        ```
        python main_DL.py --status train --model MAMTSC --if_FM 0 --training_type balance --testing_type balance --weight_path weight/notpretrain/Transformer_1yearDrop30%
        ```
    - **TCN:**
        ```
        python main_DL.py --status train --model TCN --if_FM 0 --training_type balance --testing_type balance --weight_path weight/notpretrain/TCN_1yearDrop30% --d_model 30 --batch_size 64 --learning_rate 0.004
        ```
- ML-based:
    - **Decision Tree:**
        ```
        # Last 1 Month Average
        python main_ML.py --status train --model DT --format LastTS --weight_path weight/ML/DT_LastMonthAvg --testing_type balance

        # Last 3 Month Average
        python main_ML.py --status train --model DT --format Last3TS --weight_path weight/ML/DT_Last3MonthAvg --testing_type balance

        # Flatten
        python main_ML.py --status train --model DT --format Flatten --weight_path weight/ML/DT_Flatten --testing_type balance
        ```
    - **Random Forest:**
        ```
        # Last 1 Month Average
        python main_ML.py --status train --model RF --format LastTS --weight_path weight/ML/RF_LastMonthAvg --testing_type balance

        # Last 3 Month Average
        python main_ML.py --status train --model RF --format Last3TS --weight_path weight/ML/RF_Last3MonthAvg --testing_type balance

        # Flatten
        python main_ML.py --status train --model RF --format Flatten --weight_path weight/ML/RF_Flatten --testing_type balance
        ```
    - **XGBoost:**
        ```
        # Last 1 Month Average
        python main_ML.py --status train --model XGB --format LastTS --weight_path weight/ML/XGB_LastMonthAvg --testing_type balance

        # Last 3 Month Average
        python main_ML.py --status train --model XGB --format Last3TS --weight_path weight/ML/XGB_Last3MonthAvg --testing_type balance

        # Flatten
        python main_ML.py --status train --model XGB --format Flatten --weight_path weight/ML/XGB_Flatten --testing_type balance
        ```
