import Train_Model
import Audiosource
import Predict

trainer = Train_Model("dataset.csv")
trainer.train()
trainer.save_model()



