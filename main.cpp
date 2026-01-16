#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>

// Функция для чтения CSV файла
std::vector<std::vector<double> > readCSV(const std::string &filename) {
    std::vector<std::vector<double> > data;
    std::ifstream file(filename);
    std::string line;

    while (getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }

        data.push_back(row);
    }

    return data;
}

// Функция для чтения матрицы коэффициентов логистической регрессии
std::vector<std::vector<double> > readModel(const std::string &filename) {
    return readCSV(filename);
}

// Сигмоидная функция
double sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Предсказание класса с помощью логистической регрессии
int predictLogReg(const std::vector<double> &features, const std::vector<std::vector<double> > &coefficients) {
    std::vector<double> scores(coefficients.size(), 0.0);

    // Вычисляем score для каждого класса
    for (size_t class_idx = 0; class_idx < coefficients.size(); class_idx++) {
        double score = coefficients[class_idx][0]; // Свободный член

        for (size_t i = 1; i < coefficients[class_idx].size(); ++i)
            score += coefficients[class_idx][i] * features[i - 1];

        scores[class_idx] = sigmoid(score);
    }

    // Находим класс с максимальной вероятностью
    return std::distance(scores.begin(), max_element(scores.begin(), scores.end()));
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " test.csv model.txt" << std::endl;
        return 1;
    }

    std::string testFile = argv[1];
    std::string modelFile = argv[2];

    // Чтение тестовых данных
    auto testData = readCSV(testFile);

    // Чтение модели
    auto coefficients = readModel(modelFile);

    if (coefficients.empty()) {
        std::cerr << "Error: Could not read model file" << std::endl;
        return 1;
    }

    // Проверка совместимости размерностей
    if (coefficients[0].size() != 785) {
        // 1 свободный член + 784 признака
        std::cerr << "Error: Model dimensions mismatch" << std::endl;
        return 1;
    }

    // Вычисление точности
    int correct = 0;
    auto total = testData.size();

    for (const auto &row: testData) {
        if (row.size() < 785) {
            std::cerr << "Warning: Skipping row with insufficient features" << std::endl;
            continue;
        }

        int true_label = static_cast<int>(row[0]);

        // Извлекаем признаки (пиксели)
        std::vector<double> features(row.begin() + 1, row.end());

        // Нормализация признаков от 0 до 1
        for (double &feature: features) {
            feature /= 255.0;
        }

        // Предсказание
        int predicted_label = predictLogReg(features, coefficients);

        if (predicted_label == true_label) {
            correct++;
        }
    }

    // Вывод accuracy
    double accuracy = static_cast<double>(correct) / total;
    std::cout << accuracy << std::endl;

    return 0;
}
