import random

class BOWDataset(object):

    def __init__(self, dataset=None, filename=None):

        if dataset:
            self.dataset = dataset

        elif filename:
            self.dataset = []
            file = open(filename)
            for line in file:
                self.dataset.append([ int(x) for x in str.split(line, ',') ])

        else:
            self.dataset = []

    def slice(self, min=None, max=None):

        return BOWDataset(dataset=self.dataset[min:max])

    def categories(self):

        categories = set()
        for data in self.dataset:
            categories.add(data[0])

        return categories


    def words_count(self):

        return len(self.dataset[0]) - 1


    def categories_by_word(self):

        categories_by_word = [ set() for x in range(len(self.dataset[0]) - 1) ]

        for data in self.dataset:
            for i in range(1, len(data)):
                if (data[i] > 0):
                    categories_by_word[i - 1].add(data[0])

        return categories_by_word


    def group_words_by_cat(self, max_differents_cat):

        categories_by_word = self.categories_by_word()

        new_dataset = []
        for data in self.dataset:
            words_count = 0
            new_data = [data[0]] + [0 for x in range(9)]

            for i in range(1, len(data)):
                if len(categories_by_word[i - 1]) <= max_differents_cat:
                    words_count += data[i]
                else:
                    new_data.append(data[i])

            new_data[data[0]] = words_count
            new_dataset.append(new_data)

        return BOWDataset(dataset=new_dataset)

    def shuffle(self):

        random.shuffle(self.dataset)


    def uncategorized_dataset(self):

        return  [ data[1:] for data in self.dataset ]


    def activate_neural_network(self, hnn):

        new_dataset = [ [data[0]] + hnn.activate(data[1:]).tolist() for data in self.dataset ]
        return BOWDataset(dataset=new_dataset)


    def save_to_file(filename, separator=','):

        fout = open(filename, 'w');
        for data in self.dataset:
            fout.write(separator.join(data))
        fout.close()


    def __str__(self):

        return str(self.dataset)



