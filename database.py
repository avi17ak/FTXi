import sqlite3

conn = sqlite3.connect('logins.db')
c = conn.cursor()

c.execute("""CREATE TABLE logins (
            first_name text,
            last_name text,
            username text,
            team text
            )""")

#c.execute("INSERT INTO logins VALUES ('Avkaran','Klair','13avkaranklair','Liverpool')")
c.execute("SELECT * FROM logins WHERE team='Liverpool'")
print(c.fetchone())
conn.commit()
conn.close()


 
