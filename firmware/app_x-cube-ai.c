#include "app_x-cube-ai.h"
#include "network.h"
#include "network_data.h"
#include "stai.h"
#include "main.h"
#include <stdio.h>

// Values from Colab
#define TEMP_MEAN       53.6158f
#define TEMP_STD         0.2756f
#define ANOMALY_THRESHOLD 1.309446f
#define WINDOW_SIZE      10

static stai_network network;
static float in_data[WINDOW_SIZE];
static float out_data[WINDOW_SIZE];
static stai_ptr in_ptrs[1]  = { (stai_ptr)in_data  };
static stai_ptr out_ptrs[1] = { (stai_ptr)out_data };

// Circular buffer to accumulate temperature window
static float temp_buffer[WINDOW_SIZE] = {0};
static int   buf_count = 0;

// ADC read from main.c
extern ADC_HandleTypeDef hadc3;

static float read_temperature(void)
{
    HAL_ADC_Start(&hadc3);
    HAL_ADC_PollForConversion(&hadc3, 100);
    uint32_t raw = HAL_ADC_GetValue(&hadc3);
    HAL_ADC_Stop(&hadc3);

    uint16_t *TS_CAL1 = (uint16_t*) 0x1FF1E820;
    uint16_t *TS_CAL2 = (uint16_t*) 0x1FF1E840;

    return ((110.0f - 30.0f) / (float)(*TS_CAL2 - *TS_CAL1))
           * (float)((int32_t)raw - (int32_t)*TS_CAL1) + 30.0f;
}

void MX_X_CUBE_AI_Init(void)
{
    stai_runtime_init();
    stai_network_init(&network);
    stai_network_set_inputs (&network, in_ptrs, 1);
    stai_network_set_outputs(&network, out_ptrs, 1);
}

void MX_X_CUBE_AI_Process(void)
{
    // Read one temperature sample
    float temp = read_temperature();

    // Fill circular buffer
    temp_buffer[buf_count % WINDOW_SIZE] = temp;
    buf_count++;

    // Wait until we have a full window
    if (buf_count < WINDOW_SIZE) {
        HAL_Delay(100);
        return;
    }

    // Normalize window into in_data
    for (int i = 0; i < WINDOW_SIZE; i++) {
        in_data[i] = (temp_buffer[i] - TEMP_MEAN) / TEMP_STD;
    }

    // Run inference
    stai_network_run(&network, STAI_MODE_SYNC);

    // Compute MSE
    float error = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        float diff = in_data[i] - out_data[i];
        error += diff * diff;
    }
    error /= WINDOW_SIZE;

    // Decision
    if (error > ANOMALY_THRESHOLD) {
        HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, GPIO_PIN_SET);
        printf("ANOMALY! temp=%.2f error=%.4f\r\n", temp, error);
    } else {
        HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, GPIO_PIN_RESET);
        printf("Normal.  temp=%.2f error=%.4f\r\n", temp, error);
    }

    HAL_Delay(100);
}
